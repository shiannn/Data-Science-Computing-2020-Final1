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
    #print(mtx.indptr)
    #print(np.diff(mtx.indptr))
    available_neighbors = np.diff(mtx.indptr)
    #print(available_neighbors)
    ### prob should start from zero
    prob = 1 / np.repeat(available_neighbors, available_neighbors)
    roulle_wheel = np.cumsum(prob) - np.floor(np.cumsum(prob))
    roulle_wheel[mtx.indptr[1:] - 1] = 1
    roulle_wheel = roulle_wheel - prob
    roulle_wheel[mtx.indptr[:-1]] = 0
    #print(roulle_wheel)
    shot = np.random.rand(g.vcount())
    #print(shot)
    shot_on_wheel = np.repeat(shot, available_neighbors)
    #print(shot_on_wheel)
    interval_points = shot_on_wheel < roulle_wheel
    print(interval_points)
    ### Find False -> True or left point == 1
    r_interval_points = np.roll(interval_points, -1)
    #print(r_interval_points)
    r_wheel = np.roll(roulle_wheel, -1)
    #print(r_wheel)
    #exit(0)
    false2true = np.logical_and(interval_points==False, r_interval_points==True)
    false2zero = np.logical_and(interval_points==False, r_wheel==0)
    cum_selected_neighbors = np.logical_or(false2true, false2zero)
    cum_selected_neighbors_ids = cum_selected_neighbors.nonzero()
    #print(cum_selected_neighbors_ids)
    neighbors = mtx.indices[cum_selected_neighbors_ids]
    print('neighbors.shape', neighbors.shape)
    #exit(0)
    
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