from config import karate_dataset, coauthors_dataset
import igraph
from igraph import Graph, VertexClustering
from scipy.io import mmread
import numpy as np
import random

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import time

dataset = coauthors_dataset
print(str(dataset))
#karate_set = Graph.Load(str(karate_dataset), format="edge")
#g = igraph.read(str(dataset), format='edge', directed=False)
#g.delete_vertices(g.vs[0].index)
#print(len(g.vs))
print('read mtx file')
mtx = mmread(str(dataset)).tocsr()
print('to igraph')
srcs, tgts = mtx.nonzero()
g = Graph(list(zip(srcs.tolist(), tgts.tolist())))
print(type(mtx))
print('start')

#mtx = g.get_adjacency_sparse()
#mtx.indices
#mtx.indptr
#np.split(mtx.indices, mtx.indptr[1:-1])
st = time.time()
for it in range(15):
    print(mtx.indptr)
    print(np.diff(mtx.indptr))
    chosen_part = np.random.uniform(low=0.0, high=1.0, size=np.diff(mtx.indptr).shape)
    chosen_in_each_interval = np.floor(np.diff(mtx.indptr)* chosen_part).astype(int)
    neighs = chosen_in_each_interval + mtx.indptr[:-1]
    print(neighs)
    neighbors = mtx.indices[neighs]
    print(np.arange(neighbors.shape[0]))
    row = np.arange(neighbors.shape[0])
    col = neighbors[np.arange(neighbors.shape[0])]
    print('row', row.shape, row)
    print('col', col.shape, col)
    my_g = csr_matrix((np.ones(row.shape), (row, col)), shape=(row.shape[0], col.shape[0]))
    print(my_g.shape)
    num_cluster, membership = connected_components(my_g)
    print(membership)
    score = g.modularity(membership)
    print(score)
ed = time.time()
print(ed - st)
#neighbors.shape
#mtx.indices
#print(type(mtx))
#print((mtx==1).shape)
#print(mtx[[1,3,5], [1,2,3]])
#print(mtx.nonzero().reshape(-1,2))
#print(mtx.shape)
#np.unique(mtx, axis=1)
#np.median(score)
#print(g)
#print(g.vs[5])
#for v in g.vs:
#    print(v.attributes())
#part = [a for a in range(35)]
#groups = [{34, 10, 12, 21, 23}, {32, 3, 14, 18, 19, 22, 25, 26}, {17, 33, 20, 29, 31}, {1, 2, 7, 9, 11, 13, 24, 30}, {4, 5, 6, 8, 15, 16, 27, 28}]
#part = [0]* 35
#for idx, group in enumerate(groups):
#    for p in group:
#        part[p] = idx
#part = np.random.randint(5, size=35)
#print(part)

#print(g.modularity(membership=part))
#print(g.neighbors(g.vs[0]))
"""
print(g.ecount())
sp_tree = g.spanning_tree()
print(sp_tree.ecount())
print(sp_tree.es[0].source, sp_tree.es[0].target)
print(sp_tree.es[1].source, sp_tree.es[1].target)
sp_tree.delete_edges(sp_tree.es[0])
print(sp_tree.es[0].source, sp_tree.es[0].target)
print(sp_tree.es[1].source, sp_tree.es[1].target)
"""
#for eg in sp_tree.es:
#    print(eg.source, eg.target)
"""
com = Graph(len(g.vs))
print('finish create graph')
exit(0)
for vtx in g.vs:
    #for neigh in g.neighbors(vtx):
    #print(vtx.index)
    if vtx.index == 0:
        continue
    neighs = random.sample(g.neighbors(vtx), 1)
    for neigh in neighs:
        #print(vtx.index, neigh)
        com.add_edge(vtx.index, neigh)
com.simplify()
print(com.components())
print(com.components().membership)
print(g.modularity(com.components().membership))
"""