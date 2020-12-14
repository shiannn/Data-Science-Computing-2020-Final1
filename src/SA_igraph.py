import copy
import random
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import igraph
from igraph import Graph, VertexClustering
from config import karate_dataset, coauthors_dataset, ans_dir
from pathlib import Path

from argument import parser_SA

class SimulatedAnnealing():
    def __init__(self, graph, mtx, max_iterations, min_temperature, epsilon):
        self.graph = graph
        self.mtx = mtx
        #self.pcur = np.random.randint(len(graph.nodes), size=len(graph.nodes))
        self.pcur = Solution(graph, mtx)
        #print(self.pcur.x)
        #print(self.pcur.y)
        self.pnew = copy.deepcopy(self.pcur)
        self.pb = None
        self.max_iterations = max_iterations
        self.temperature = np.inf
        self.min_temperature = min_temperature
        self.Tstart = self.init_temperature()
        self.epsilon = epsilon

    
    def init_temperature(self):
        return 300

    def simulated_annealing(self):
        self.pcur.y = self.f_objective_function(self.pcur.x)
        #print(self.pcur.y)
        self.pb = copy.deepcopy(self.pcur)
        iteration = 0
        while(not self.c_termination(iteration, self.temperature)):
            self.pnew.x = self.mutation(self.pcur.x)
            self.pnew.y = self.f_objective_function(self.pnew.x)
            #print(self.pnew.y)
            dE = -(self.pnew.y - self.pcur.y)
            #print(iteration, self.pcur.x, self.pcur.y)
            #print(dE, self.pnew.y, self.pcur.y)
            if dE <= 0:
                self.pcur = copy.deepcopy(self.pnew)
                if self.pcur.y > self.pb.y:
                    self.pb = copy.deepcopy(self.pcur)
            else:
                self.temperature = self.getTemp(iteration)
                if np.random.rand() < np.exp(-dE/self.temperature):
                    self.pcur = copy.deepcopy(self.pnew)

            iteration += 1
        
        print('simulated annealing', self.pb.x, self.pb.y)
        return self.pb
    
    def getTemp(self, iteration):
        decay = (1 - self.epsilon)**iteration
        return (decay* self.Tstart)

    def mutation(self, x):
        ### x should be a locus_based vector (num_vertices) with value as chosen neighbor
        #print(x)
        start_point = self.mtx.indptr
        #print('st',st)
        num_avail_neighbors = np.diff(self.mtx.indptr)
        #print('num_avail_neighbors', num_avail_neighbors)
        chosen_neighbor_id = x - num_avail_neighbors
        chosen_neighbor_id = np.mod(chosen_neighbor_id, num_avail_neighbors)
        #print('chosen_neigh', chosen_neighbor_id)
        #print('chosen_neigh mode', np.mod(chosen_neighbor_id, num_avail_neighbors))
        mut_pos = np.random.randint(self.graph.vcount(), size=self.graph.vcount()//2)
        #print(mut_pos)
        chosen_neighbor_id[mut_pos] += 1
        chosen_neighbor_id = np.mod(chosen_neighbor_id, num_avail_neighbors)
        #print(chosen_neighbor_id)
        neighbor_pos = chosen_neighbor_id + self.mtx.indptr[:-1]
        locus_based = self.mtx.indices[neighbor_pos]
        #print(locus_based)
        
        return locus_based
    
    def c_termination(self, iteration, temperature):
        if iteration >= self.max_iterations:
            return True
        if self.temperature <= self.min_temperature:
            return True

    def f_objective_function(self, x):
        ### x should be a locus_based vector (num_vertices) with value as chosen neighbor
        row = np.arange(x.shape[0])
        col = x[row]
        attempt_graph = csr_matrix((np.ones(row.shape), (row, col)), shape=(row.shape[0], col.shape[0]))
        num_cluster, membership = connected_components(attempt_graph)
        score = self.graph.modularity(membership)
        
        return score
    
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
    #graph = nx.from_scipy_sparse_matrix(kara_set)
    #ret = treetraverse(graph)
    #pso = ParticleSwarm(graph = graph, N_swarm_size=5, D_dimension=len(graph.nodes), c1=1, c2=1, max_iterations=10)
    sa = SimulatedAnnealing(graph, mtx, 
        args.iterations, args.min_temperature, args.epsilon
    )
    
    best_sol = sa.simulated_annealing()
    best_value = sa.f_objective_function(best_sol.x)
    best_locus = best_sol.x
    num_cluster, membership = sa.get_membership(best_locus)
    print(membership)
    print(best_value)

    if args.no_save:
        pass
    else:
        name = 'SA_{}_{}.npy'.format(dataset.stem, np.round(best_value, 4))
        np.save(ans_dir / Path(name), membership)
    

if __name__ == '__main__':
    args = parser_SA()
    main(args)