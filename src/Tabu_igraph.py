import copy
import numpy as np
import random
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import igraph
from igraph import Graph, VertexClustering
from config import karate_dataset, coauthors_dataset

class TabuSearch():
    def __init__(self, graph, mtx):
        self.graph = graph
        self.mtx = mtx

        self.pcur = Solution(graph, mtx)
        self.pbest = copy.deepcopy(self.pcur)
        #self.pnew = copy.copy(self.pcur)
        #self.ptest = copy.copy(self.pcur)
        self.pnew = None
        self.ptest = None
        self.tabu = []
        self.max_tabu_len = 15
        self.max_iterations = 10

        ### moves will be delete the connection
        self.moves = np.arange(self.pcur.x.shape[0])

        self.plot_list = []
    
    def tabu_search(self):
        #self.pbest.x = self.initial_x()
        #self.pbest.x = self.initial_x_with_randomWalk()
        self.pcur.y = self.f_objective_function(self.pcur.x)
        self.pbest.y = self.f_objective_function(self.pbest.x)
        #self.pcur = copy.copy(self.pbest)
        self.tabu = []
        self.plot_list = []

        iteration = 0
        while(not self.c_termination(iteration)):#or self.pcur
            iteration += 1
            self.pnew = None
            moveb = None
            #print(self.pcur.x, self.pcur.y)
            for midx, move in enumerate(np.random.choice(self.moves, size=min(self.moves.shape[0], 5000))):
                tmp_pcur = copy.deepcopy(self.pcur)
                self.ptest = self.mutation(tmp_pcur, move)
                
                self.ptest.y = self.f_objective_function(self.ptest.x)
                COND = ((move not in self.tabu) and (((self.pnew is None) or (self.ptest.y > self.pnew.y)) or self.ptest.y > self.pbest.y))
                if COND:
                    self.pnew = copy.deepcopy(self.ptest)
                    moveb = move

                if midx % 500 == 0:
                    if self.pnew is not None:
                        print(self.pnew.y)
                    print(midx, self.ptest.y)
            #if iteration == 3:
            #    exit(0)
            self.pcur = copy.deepcopy(self.pnew)
            if self.pcur:
                if self.pcur.y >= self.pbest.y:
                    self.pbest = copy.deepcopy(self.pcur)
                ### inverse?
                self.tabu.append(moveb)
                if len(self.tabu) >= self.max_tabu_len:
                    self.tabu.pop(0)
            
            #print(self.pnew.disjointSet.edges, self.pnew.y)
            print(self.pbest.y, self.tabu)
            #print(self.pbest.x, self.pbest.y, self.tabu)
            self.plot_list.append(self.pbest.y)
            #exit(0)
            
        return self.pbest

    def initial_x_with_randomWalk(self):
        ret = self.randomWalk.random_walk()
        #print('random walk', ret.x, ret.y)
        return ret.x

    def initial_x(self):
        return tuple(range(self.distance_matrix.shape[1]))

    def mutation(self, pcur, move):
        ### substitution as -1
        #print(move)
        pcur.x[move] = -1
        
        return pcur

    def f_objective_function(self, x):
        ### x should be a locus_based vector (num_vertices) with value as chosen neighbor
        row = np.arange(x[x!=-1].shape[0])
        col = x[x!=-1]
        attempt_graph = csr_matrix((np.ones(row.shape), (row, col)), shape=(x.shape[0], x.shape[0]))
        num_cluster, membership = connected_components(attempt_graph)
        score = self.graph.modularity(membership)
        
        return score

    def c_termination(self, iteration):
        return (iteration >= self.max_iterations)

class Solution():
    def __init__(self, graph, mtx):
        num_avail_neighbors = np.diff(mtx.indptr)
        chosen_nums = np.random.randint(graph.vcount(), size = num_avail_neighbors.shape)
        chosen_neighbor_ids = np.mod(chosen_nums, num_avail_neighbors)#.astype(int)
        neighs_pos = chosen_neighbor_ids + mtx.indptr[:-1]
        locus_based = mtx.indices[neighs_pos]

        row = np.arange(locus_based.shape[0])
        col = locus_based[np.arange(locus_based.shape[0])]
        self.x = locus_based
        self.y = None
        print(self.x)

def main():
    #dataset = karate_dataset
    dataset = coauthors_dataset
    print('read mtx file')
    mtx = mmread(str(dataset)).tocsr()
    print('to igraph')
    srcs, tgts = mtx.nonzero()
    graph = Graph(list(zip(srcs.tolist(), tgts.tolist())))
    print('start')
    tabuSearch = TabuSearch(graph, mtx)
    #print(tabuSearch.moves)
    
    tabuSearch.tabu_search()

if __name__ == '__main__':
    import networkx as nx
    from scipy.io import mmread
    import networkx.algorithms.community as nx_comm
    from config import karate_dataset
    main()