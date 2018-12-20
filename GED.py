import numpy as np
from scipy.optimize import linear_sum_assignment

def graph_edit_distance(G1, G2):
    bestcost = None
    for vertex_path, edge_path, cost in \
        optimize_edit_paths(G1, G2):
        #assert bestcost is None or cost < bestcost
        bestcost = cost
        vert = vertex_path
        edge = edge_path
    print(vert)
    print(edge)
    return bestcost

def optimize_edit_paths(G1, G2):
    class CostMatrix:
        def __init__(self, C, lsa_row_ind, lsa_col_ind, ls):
            self.C = C
            self.lsa_row_ind = lsa_row_ind
            self.lsa_col_ind = lsa_col_ind
            self.ls = ls

    def make_CostMatrix(C, m, n):
        lsa_row_ind, lsa_col_ind = linear_sum_assignment(C)

        indexes = zip(range(len(lsa_row_ind)), lsa_row_ind, lsa_col_ind)
        subst_ind = list(k for k, i, j in indexes if i < m and j < n)
        indexes = zip(range(len(lsa_row_ind)), lsa_row_ind, lsa_col_ind)
        dummy_ind = list(k for k, i, j in indexes if i >= m and j >= n)

        lsa_row_ind[dummy_ind] = lsa_col_ind[subst_ind] + m
        lsa_col_ind[dummy_ind] = lsa_row_ind[subst_ind] + n

        return CostMatrix(C, lsa_row_ind, lsa_col_ind,
                          C[lsa_row_ind, lsa_col_ind].sum())

    def extract_C(C, i, j, m, n):

        row_ind = [k in i or k - m in j for k in range(m + n)]
        col_ind = [k in j or k - n in i for k in range(m + n)]
        return C[row_ind, :][:, col_ind]

    def reduce_C(C, i, j, m, n):

        row_ind = [k not in i and k - m not in j for k in range(m + n)]
        col_ind = [k not in j and k - n not in i for k in range(m + n)]
        return C[row_ind, :][:, col_ind]

    def reduce_ind(ind, i):

        rind = ind[[k not in i for k in ind]]
        for k in set(i):
            rind[rind >= k] -= 1
        return rind

    def match_edges(u, v, pending_g, pending_h, Ce, matched_uv=[]):
        """
        Parameters:
            u, v: matched vertices, u=None or v=None for
               deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_uv: partial vertex edit path
                list of tuples (u, v) of previously matched vertex
                    mappings u<->v, u=None or v=None for
                    deletion/insertion
        Returns:
            list of (i, j): indices of edge mappings g<->h
            localCe: local CostMatrix of edge mappings
                (basically submatrix of Ce at cross of rows i, cols j)
        """
        M = len(pending_g)
        N = len(pending_h)
        #assert Ce.C.shape == (M + N, M + N)

        g_ind = [i for i in range(M) if pending_g[i][:2] == (u, u) or
                 any(pending_g[i][:2] in ((p, u), (u, p))
                     for p, q in matched_uv)]
        h_ind = [j for j in range(N) if pending_h[j][:2] == (v, v) or
                 any(pending_h[j][:2] in ((q, v), (v, q))
                     for p, q in matched_uv)]
        m = len(g_ind)
        n = len(h_ind)

        if m or n:
            C = extract_C(Ce.C, g_ind, h_ind, M, N)
            #assert C.shape == (m + n, m + n)

            # Forbid structurally invalid matches
            # NOTE: inf remembered from Ce construction
            for k, i in zip(range(m), g_ind):
                g = pending_g[i][:2]
                for l, j in zip(range(n), h_ind):
                    h = pending_h[j][:2]
                    if nx.is_directed(G1) or nx.is_directed(G2):
                        if any(g == (p, u) and h == (q, v) or
                               g == (u, p) and h == (v, q)
                               for p, q in matched_uv):
                            continue
                    else:
                        if any(g in ((p, u), (u, p)) and h in ((q, v), (v, q))
                               for p, q in matched_uv):
                            continue
                    if g == (u, u):
                        continue
                    if h == (v, v):
                        continue
                    C[k, l] = inf

            localCe = make_CostMatrix(C, m, n)
            ij = list((g_ind[k] if k < m else M + h_ind[l],
                       h_ind[l] if l < n else N + g_ind[k])
                      for k, l in zip(localCe.lsa_row_ind, localCe.lsa_col_ind)
                      if k < m or l < n)

        else:
            ij = []
            localCe = CostMatrix(np.empty((0, 0)), [], [], 0)

        return ij, localCe

    def reduce_Ce(Ce, ij, m, n):
        if len(ij):
            i, j = zip(*ij)
            m_i = m - sum(1 for t in i if t < m)
            n_j = n - sum(1 for t in j if t < n)
            return make_CostMatrix(reduce_C(Ce.C, i, j, m, n), m_i, n_j)
        else:
            return Ce

    def get_edit_ops(matched_uv, pending_u, pending_v, Cv,
                     pending_g, pending_h, Ce, matched_cost):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path
        Returns:
            sequence of
                (i, j): indices of vertex mapping u<->v
                Cv_ij: reduced CostMatrix of pending vertex mappings
                    (basically Cv with row i, col j removed)
                list of (x, y): indices of edge mappings g<->h
                Ce_xy: reduced CostMatrix of pending edge mappings
                    (basically Ce with rows x, cols y removed)
                cost: total cost of edit operation
            NOTE: most promising ops first
        """
        m = len(pending_u)
        n = len(pending_v)
        #assert Cv.C.shape == (m + n, m + n)

        # 1) a vertex mapping from optimal linear sum assignment
        i, j = min((k, l) for k, l in zip(Cv.lsa_row_ind, Cv.lsa_col_ind)
                   if k < m or l < n)
        xy, localCe = match_edges(pending_u[i] if i < m else None,
                                  pending_v[j] if j < n else None,
                                  pending_g, pending_h, Ce, matched_uv)
        Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
        #assert Ce.ls <= localCe.ls + Ce_xy.ls
        if prune(matched_cost + Cv.ls + localCe.ls + Ce_xy.ls):
            pass
        else:
            # get reduced Cv efficiently
            Cv_ij = CostMatrix(reduce_C(Cv.C, (i,), (j,), m, n),
                               reduce_ind(Cv.lsa_row_ind, (i, m + j)),
                               reduce_ind(Cv.lsa_col_ind, (j, n + i)),
                               Cv.ls - Cv.C[i, j])
            yield (i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls

        # 2) other candidates, sorted by lower-bound cost estimate
        other = list()
        fixed_i, fixed_j = i, j
        if m <= n:
            candidates = ((t, fixed_j) for t in range(m + n)
                          if t != fixed_i and (t < m or t == m + fixed_j))
        else:
            candidates = ((fixed_i, t) for t in range(m + n)
                          if t != fixed_j and (t < n or t == n + fixed_i))
        for i, j in candidates:
            if prune(matched_cost + Cv.C[i, j] + Ce.ls):
                continue
            Cv_ij = make_CostMatrix(reduce_C(Cv.C, (i,), (j,), m, n),
                                    m - 1 if i < m else m,
                                    n - 1 if j < n else n)
            #assert Cv.ls <= Cv.C[i, j] + Cv_ij.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + Ce.ls):
                continue
            xy, localCe = match_edges(pending_u[i] if i < m else None,
                                      pending_v[j] if j < n else None,
                                      pending_g, pending_h, Ce, matched_uv)
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls):
                continue
            Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))
            #assert Ce.ls <= localCe.ls + Ce_xy.ls
            if prune(matched_cost + Cv.C[i, j] + Cv_ij.ls + localCe.ls +
                     Ce_xy.ls):
                continue
            other.append(((i, j), Cv_ij, xy, Ce_xy, Cv.C[i, j] + localCe.ls))

        # yield from
        for t in sorted(other, key=lambda t: t[4] + t[1].ls + t[3].ls):
            yield t

    def get_edit_paths(matched_uv, pending_u, pending_v, Cv,
                       matched_gh, pending_g, pending_h, Ce, matched_cost):
        """
        Parameters:
            matched_uv: partial vertex edit path
                list of tuples (u, v) of vertex mappings u<->v,
                u=None or v=None for deletion/insertion
            pending_u, pending_v: lists of vertices not yet mapped
            Cv: CostMatrix of pending vertex mappings
            matched_gh: partial edge edit path
                list of tuples (g, h) of edge mappings g<->h,
                g=None or h=None for deletion/insertion
            pending_g, pending_h: lists of edges not yet mapped
            Ce: CostMatrix of pending edge mappings
            matched_cost: cost of partial edit path
        Returns:
            sequence of (vertex_path, edge_path, cost)
                vertex_path: complete vertex edit path
                    list of tuples (u, v) of vertex mappings u<->v,
                    u=None or v=None for deletion/insertion
                edge_path: complete edge edit path
                    list of tuples (g, h) of edge mappings g<->h,
                    g=None or h=None for deletion/insertion
                cost: total cost of edit path
            NOTE: path costs are non-increasing
        """
 
        if prune(matched_cost + Cv.ls + Ce.ls):
            return

        if not max(len(pending_u), len(pending_v)):
            #assert not len(pending_g)
            #assert not len(pending_h)
            # path completed!
            #assert matched_cost <= maxcost.value
            maxcost.value = min(maxcost.value, matched_cost)
            yield matched_uv, matched_gh, matched_cost

        else:
            edit_ops = get_edit_ops(matched_uv, pending_u, pending_v, Cv,
                                    pending_g, pending_h, Ce, matched_cost)
            for ij, Cv_ij, xy, Ce_xy, edit_cost in edit_ops:
                i, j = ij
                #assert Cv.C[i, j] + sum(Ce.C[t] for t in xy) == edit_cost
                if prune(matched_cost + edit_cost + Cv_ij.ls + Ce_xy.ls):
                    continue

                # dive deeper
                u = pending_u.pop(i) if i < len(pending_u) else None
                v = pending_v.pop(j) if j < len(pending_v) else None
                matched_uv.append((u, v))
                for x, y in xy:
                    len_g = len(pending_g)
                    len_h = len(pending_h)
                    matched_gh.append((pending_g[x] if x < len_g else None,
                                       pending_h[y] if y < len_h else None))
                sortedx = list(sorted(x for x, y in xy))
                sortedy = list(sorted(y for x, y in xy))
                G = list((pending_g.pop(x) if x < len(pending_g) else None)
                         for x in reversed(sortedx))
                H = list((pending_h.pop(y) if y < len(pending_h) else None)
                         for y in reversed(sortedy))

                # yield from
                for t in get_edit_paths(matched_uv, pending_u, pending_v,
                                        Cv_ij,
                                        matched_gh, pending_g, pending_h,
                                        Ce_xy,
                                        matched_cost + edit_cost):
                    yield t

                # backtrack
                if u is not None:
                    pending_u.insert(i, u)
                if v is not None:
                    pending_v.insert(j, v)
                matched_uv.pop()
                for x, g in zip(sortedx, reversed(G)):
                    if g is not None:
                        pending_g.insert(x, g)
                for y, h in zip(sortedy, reversed(H)):
                    if h is not None:
                        pending_h.insert(y, h)
                for t in xy:
                    matched_gh.pop()

    # Initialization

    pending_u = list(G1.nodes)
    pending_v = list(G2.nodes)

    # cost matrix of vertex mappings
    m = len(pending_u)
    n = len(pending_v)
    C = np.zeros((m + n, m + n))
    
    inf = n + m + 1
    C[0:m, n:n + m] = np.array([1 if i == j else inf
                                for i in range(m) for j in range(m)]
                               ).reshape(m, m)
    C[m:m + n, 0:n] = np.array([1 if i == j else inf
                                for i in range(n) for j in range(n)]
                               ).reshape(n, n)
    Cv = make_CostMatrix(C, m, n)

    pending_g = list(G1.edges)
    pending_h = list(G2.edges)

    m = len(pending_g)
    n = len(pending_h)
    C = np.zeros((m + n, m + n))
    
    inf = n + m + 1
    C[0:m, n:n + m] = np.array([1 if i == j else inf
                                for i in range(m) for j in range(m)]
                               ).reshape(m, m)
    C[m:m + n, 0:n] = np.array([1 if i == j else inf
                                for i in range(n) for j in range(n)]
                               ).reshape(n, n)
    Ce = make_CostMatrix(C, m, n)


    class MaxCost:
        def __init__(self):
            self.value = Cv.C.sum() + Ce.C.sum() + 1
    maxcost = MaxCost()

    def prune(cost):
        if cost >= maxcost.value:
            return True

    for vertex_path, edge_path, cost in \
        get_edit_paths([], pending_u, pending_v, Cv,
                       [], pending_g, pending_h, Ce, 0):

        yield list(vertex_path), list(edge_path), cost
