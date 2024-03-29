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

from argument import parser_PSO

class ParticleSwarm():
    def __init__(self, graph, mtx, lower_bound=0, upper_bound=1, max_iterations=20, N_swarm_size=4, D_dimension=34, 
        c1=1, c2=1, w=1, swarm_position=None, swarm_velocity=None):

        self.graph = graph
        self.mtx = mtx
        self.swarm_size = N_swarm_size
        self.dimension = D_dimension

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if swarm_position is None:
            self.swarm_position = self.lower_bound + (self.upper_bound - self.lower_bound)* np.random.rand(
                self.swarm_size, self.dimension
            )
            #self.initial_position()
        else:
            self.swarm_position = position

        self.swarm_velocity = np.zeros((self.swarm_size, self.dimension))
        #self.swarm_velocity = np.random.rand(self.swarm_size, self.dimension)

        self.local_best = ((-np.inf)* np.ones(self.swarm_size)).squeeze()
        self.global_best = -np.inf
        self.local_center = ((-np.inf)* np.ones((self.swarm_size, self.dimension))).squeeze()
        self.global_center = -np.inf

        self.c1 = c1
        self.c2 = c2
        self.w = w

        self.max_iterations = max_iterations
        #print(self.graph.subgraph([1,20,33]).edges)
    
    def initial_position(self):
        num_avail_neighbors = np.diff(self.mtx.indptr)
        neighbor_degrees = np.array(self.graph.degree(self.mtx.indices))
        for idx, _ in enumerate(self.mtx.indptr[:-1]):
            #print(self.mtx.indptr[idx])
            #print(neighbor_degrees[self.mtx.indptr[idx]:self.mtx.indptr[idx+1]])
            neign_degrees = neighbor_degrees[self.mtx.indptr[idx]:self.mtx.indptr[idx+1]]
            max_degree_neighbor = np.argsort(neign_degrees)[-1]
            #print(max_degree_neighbor)
            self.swarm_position[:self.swarm_size//3, idx] = max_degree_neighbor / num_avail_neighbors[idx]
        #print(self.swarm_position)

    def particleSwarm(self):
        for iteration in range(self.max_iterations):
            if self.converge(self.swarm_position):
                break
            print(iteration)
            #print(self.swarm_position.shape)
            encountered_by_particles = self.f_objective_function(self.swarm_position)
            #print(encountered_by_particles)
            #exit(0)

            #self.local_center = np.where(encountered_by_particles > self.local_best, self.swarm_position, self.local_center)
            self.local_center[encountered_by_particles > self.local_best] = self.swarm_position[encountered_by_particles > self.local_best]
            self.local_best = np.where(
                encountered_by_particles > self.local_best, encountered_by_particles, self.local_best
            )
            self.global_center = self.swarm_position[np.argmax(encountered_by_particles)] if np.max(encountered_by_particles) > self.global_best else self.global_center
            self.global_best = max(self.global_best, np.max(encountered_by_particles))
            print(self.global_best)

            #local_effect = np.diag(self.local_center - self.swarm_position).reshape(-1,1)
            local_effect = self.local_center - self.swarm_position
            global_effect = self.global_center - self.swarm_position
            #print(local_effect.shape)
            #print(global_effect.shape)
            #exit(0)
            
            self.swarm_velocity = (self.w* self.swarm_velocity +
                self.c1* np.random.rand()* local_effect +
                self.c2* np.random.rand()* global_effect
            )

            self.swarm_position = self.swarm_position + self.swarm_velocity
            ### trim self.swarm_position
            #self.swarm_position[self.swarm_position>1] -= 2*(self.swarm_position[self.swarm_position>1]-1)
            #self.swarm_position[self.swarm_position<0] *= -1
            #self.swarm_position -= np.floor(self.swarm_position)
            self.swarm_position[self.swarm_position>1] = 0.99999
            self.swarm_position[self.swarm_position<0] = 0

            #self.w = self.w_max - (self.w_max - self.w_min)* iteration / self.max_iterations

        #print('ans', self.global_center, self.global_best)
        return self.global_center, self.global_best

    def converge(self, position):
        score = self.f_objective_function(position)
        
        #return (values == values[0]).all()
        return (score > 0.8).any()
        
    def f_objective_function(self, x):
        scores = []
        for particle in x:
            locus_based = self.get_locus(particle)
            #print(locus_based)
            num_cluster, membership = self.get_membership(locus_based)
            u, cnts = np.unique(membership, return_counts=True)
            constrain_coeff = 0.5 if self.graph.vcount() < 100 else 0.1
            if cnts.max() > constrain_coeff* self.graph.vcount():
                score = 0
            else:
                score = self.graph.modularity(membership)
            scores.append(score)
            #print(score)
        scores = np.array(scores)
        #print(scores)
        
        return scores
    
    def get_membership(self, locus_based):
        row = np.arange(locus_based.shape[0])
        col = locus_based[row]
        attempt_graph = csr_matrix((np.ones(row.shape), (row, col)), shape=(row.shape[0], col.shape[0]))
        num_cluster, membership = connected_components(attempt_graph)

        return num_cluster, membership

    def get_locus(self, x):
        num_avail_neighbors = np.diff(self.mtx.indptr)
        #chosen_neighbor_ids = np.floor(num_avail_neighbors* x).astype(int)
        chosen_neighbor_ids = np.mod(x*1000000, num_avail_neighbors).astype(int)
        neighs_pos = chosen_neighbor_ids + self.mtx.indptr[:-1]
        locus_based = self.mtx.indices[neighs_pos]

        return locus_based

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
    #pso = ParticleSwarm(graph = graph, N_swarm_size=5, D_dimension=len(graph.nodes), c1=1, c2=1, max_iterations=10)
    pso = ParticleSwarm(
        graph, mtx, N_swarm_size=args.popu_size, D_dimension=graph.vcount(), 
        c1=args.local_effect, c2=args.global_effect, max_iterations=args.iterations
    )
    
    global_center, global_best = pso.particleSwarm()
    best_locus = pso.get_locus(global_center)
    num_cluster, membership = pso.get_membership(best_locus)

    if args.no_save:
        pass
    else:
        if global_best > 0.4:
            name = 'PSO_{}_{}.npy'.format(dataset.stem, np.round(global_best, 4))
            np.save(ans_dir / Path(name), membership)

if __name__ == '__main__':
    args = parser_PSO()
    main(args)