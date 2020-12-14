import copy
import numpy as np
import random
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import igraph
from igraph import Graph, VertexClustering
from config import karate_dataset, coauthors_dataset, ans_dir

from argument import parser_Tabu

class TabuSearch():
    def __init__(self, graph, mtx, max_tabu_len, max_iterations):
        self.graph = graph
        self.mtx = mtx

        self.pcur = Solution(graph, mtx)
        self.pbest = copy.deepcopy(self.pcur)
        self.ptest = copy.deepcopy(self.pcur)
        #self.pnew = copy.copy(self.pcur)
        self.pnew = None
        self.tabu = []
        self.max_tabu_len = max_tabu_len
        self.max_iterations = max_iterations

        ### moves will change its neighbor
        self.num_avail_neighbors = np.diff(self.mtx.indptr)
        self.moves = [(v, self.num_avail_neighbors[v]) for v in np.arange(self.graph.vcount())]
        print(self.moves)

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
            for midx, move in enumerate(self.moves):
                tmp_pcur = copy.deepcopy(self.pcur)
                self.ptest.x = self.mutation(tmp_pcur, move)
                
                self.ptest.y = self.f_objective_function(self.ptest.x)
                COND = ((move not in self.tabu) and (((self.pnew is None) or (self.ptest.y > self.pnew.y)) or self.ptest.y > self.pbest.y))
                if COND:
                    self.pnew = copy.deepcopy(self.ptest)
                    moveb = move

            self.pcur = copy.deepcopy(self.pnew)
            if self.pcur:
                if self.pcur.y >= self.pbest.y:
                    self.pbest = copy.deepcopy(self.pcur)
                ### inverse?
                self.tabu.append(moveb)
                if len(self.tabu) >= self.max_tabu_len:
                    self.tabu.pop(0)
            
            #print(self.pbest.y, self.tabu)
            #print(self.pbest.x, self.pbest.y, self.tabu)
            self.plot_list.append(self.pbest.y)
            #exit(0)
            
        return self.pbest

    def mutation(self, pcur, move):
        ### substitution as -1
        pcur.x[move[0]] = self.mtx.indices[self.mtx.indptr[move[0]] + move[1] - 1]
        locus_based = pcur.x
        """
        start_point = self.mtx.indptr
        #print('st',st)
        
        #print('num_avail_neighbors', num_avail_neighbors)
        chosen_neighbor_id = self.pcur.x - num_avail_neighbors
        chosen_neighbor_id = np.mod(chosen_neighbor_id, num_avail_neighbors)
        #print('chosen_neigh', chosen_neighbor_id)
        #print('chosen_neigh mode', np.mod(chosen_neighbor_id, num_avail_neighbors))
        mut_pos = np.random.randint(self.graph.vcount())
        #print(mut_pos)
        chosen_neighbor_id[move] += 1
        chosen_neighbor_id = np.mod(chosen_neighbor_id, num_avail_neighbors)
        #print(chosen_neighbor_id)
        neighbor_pos = chosen_neighbor_id + self.mtx.indptr[:-1]
        locus_based = self.mtx.indices[neighbor_pos]
        """
        
        return locus_based

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
    
    def get_membership(self, locus_based):
        row = np.arange(locus_based.shape[0])
        col = locus_based[row]
        attempt_graph = csr_matrix((np.ones(row.shape), (row, col)), shape=(row.shape[0], col.shape[0]))
        num_cluster, membership = connected_components(attempt_graph)

        return num_cluster, membership

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

def main(args):
    print(args)
    if args.datasets == 'karate':
        dataset = karate_dataset
    else:
        dataset = coauthors_dataset

    print('read mtx file')
    mtx = mmread(str(dataset)).tocsr()
    print('to igraph')
    srcs, tgts = mtx.nonzero()
    graph = Graph(list(zip(srcs.tolist(), tgts.tolist())))
    print('start')
    tabuSearch = TabuSearch(graph, mtx, 
        max_tabu_len=args.max_tabu_len, max_iterations = args.iterations
    )
    #print(tabuSearch.moves)
    
    best_sol = tabuSearch.tabu_search()
    best_value = tabuSearch.f_objective_function(best_sol.x)
    best_locus = best_sol.x
    num_cluster, membership = tabuSearch.get_membership(best_locus)
    print(best_value, membership)
    exit(0)

    if args.no_save:
        pass
    else:
        name = 'Tabu_{}_{}.npy'.format(dataset.stem, np.round(best_value, 4))
        np.save(ans_dir / Path(name), membership)

if __name__ == '__main__':
    args = parser_Tabu()
    main(args)