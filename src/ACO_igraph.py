import copy
import random
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import igraph
from igraph import Graph, VertexClustering
from config import karate_dataset, coauthors_dataset

from argument import parser_ACO

class AntColonyOptimization():
    def __init__(self, mtx, graph, N_particle_size=5, max_iterations=30, p_evaporates=0.5, alpha=1, eta_scale_parameter=2):
        self.graph = graph
        self.mtx = mtx

        self.N_particle_size = N_particle_size
        self.max_iterations = max_iterations

        self.p_evaporates = p_evaporates
        self.alpha = alpha
        self.eta_scale_parameter = eta_scale_parameter

        #self.pheromone_matrix = (1.0/(self.node_num**2))* np.ones((self.node_num, self.node_num))
        #self.pheromone_matrix = np.ones((self.node_num, self.node_num))
        #print(self.mtx.indptr.shape)
        self.num_available_neighbors = np.diff(self.mtx.indptr)
        self.pheromone_array = self.initial_pheromone()
        
        print(self.pheromone_array.shape)
        self.prob_array = self.pheromone2prob()
        print(self.prob_array.shape)

        self.best_cost = None
        self.best_solution = None
        
        self.plot_List = []
    
    def initial_pheromone(self):
        #pheromone_array = 1 / np.repeat(self.num_available_neighbors, self.num_available_neighbors)
        neighbor_degrees = np.array(self.graph.degree(self.mtx.indices))
        pheromone_array = neighbor_degrees.astype(np.float64)
        #print(pheromone_array.shape)
        
        return pheromone_array

    def pheromone2prob(self):
        ### Sum total neighbors choice pheromone for each node
        #print(self.pheromone_array)
        cum_pheromone = np.cumsum(self.pheromone_array)
        #print(cum_pheromone)
        
        sum_each_node = cum_pheromone[self.mtx.indptr[1:] - 1] - cum_pheromone[self.mtx.indptr[:-1]]
        sum_each_node += self.pheromone_array[self.mtx.indptr[:-1]]
        #print(sum_each_node)
        ### Repeat total pheromone
        rep_total_phromone = np.repeat(sum_each_node, self.num_available_neighbors)
        ### Divide total pheromone on each pheromone to get prob
        #print(rep_total_phromone[:16])
        prob = self.pheromone_array / rep_total_phromone
        
        return prob

    def antColonyOptimization(self):
        ants = [Ant(self) for _ in range(self.N_particle_size)]
        for iteration in range(self.max_iterations):
            for ant in ants:
                #for i in range(self.distance_matrix.shape[0] - 1):
                ant.move()
                #print(ant.tabu)
                #ant.total_value += self.distance_matrix[ant.tabu[0], ant.tabu[-1]]
                #print(self.f_objective_function(ant.tabu))
                print(ant.score)
                if self.best_cost is None or ant.score > self.best_cost:
                    self.best_cost = ant.score
                    self.best_solution = ant.current_state.copy()

            print('best_cost', self.best_cost)

            #self.plot_List.append(self.best_cost)

            ### the ant finish constructing its solution
            ### add pheromone on its trail
            self.update_pheromone(ants)

        
    def update_pheromone(self, ants):
        ### Get max score and min score
        pheromone_worst = max([ant.score for ant in ants])
        pheromone_best = min([ant.score for ant in ants])
        #print(pheromone_best, pheromone_worst)
        ### Calculate how many pheromone to add (the more best ants, the more added)
        
        sum_delta_pheromone = self.eta_scale_parameter*(pheromone_best - pheromone_worst)
        #print(sum_delta_pheromone)
        ### evaporate
        self.pheromone_array *= (1 - self.p_evaporates)
        ### best ant / worst ant and add on edges of best solution
        for ant in ants:
            #if ant.score <= pheromone_worst:
            #    self.pheromone_array[ant.new_neighbors_id] -= sum_delta_pheromone
            if ant.score >= self.best_cost:
                ### add pheromone on its trail if it is the best
                #self.pheromone_array[ant.new_neighbors_id] += sum_delta_pheromone
                self.pheromone_array[ant.new_neighbors_id] *= 1.01
        

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
            

class Ant():
    def __init__(self, aco):
        self.aco = aco
        self.score = 0

        self.new_neighbors_id = None
        self.current_state = None

    def move(self):
        ### calculate probability from neighborhood
        prob_array = self.aco.pheromone2prob()
        self.new_neighbors_id = self.prob2neighbor(prob_array)
        new_neighbors = self.aco.mtx.indices[self.new_neighbors_id]
        self.current_state = new_neighbors
        self.score = self.aco.f_objective_function(self.current_state)
    
    def prob2neighbor(self, prob):
        ### get roulle_wheel
        roulle_wheel = np.cumsum(prob) - np.floor(np.cumsum(prob))
        roulle_wheel[self.aco.mtx.indptr[1:] - 1] = 1
        roulle_wheel = roulle_wheel - prob
        roulle_wheel[self.aco.mtx.indptr[:-1]] = 0
        ### shot on roulle_wheel
        shot = np.random.rand(self.aco.graph.vcount())
        shot_on_wheel = np.repeat(shot, self.aco.num_available_neighbors)
        interval_points = shot_on_wheel < roulle_wheel
        ### Find False -> True or left point == 1
        r_interval_points = np.roll(interval_points, -1)
        r_wheel = np.roll(roulle_wheel, -1)
        false2true = np.logical_and(interval_points==False, r_interval_points==True)
        false2zero = np.logical_and(interval_points==False, r_wheel==0)
        ### get neighbors
        cum_selected_neighbors = np.logical_or(false2true, false2zero)
        cum_selected_neighbors_ids = cum_selected_neighbors.nonzero()
        #neighbors = self.aco.mtx.indices[cum_selected_neighbors_ids]

        return cum_selected_neighbors_ids

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
    
    aco = AntColonyOptimization(
        mtx, graph, N_particle_size=args.popu_size,
        p_evaporates=args.evaporate, eta_scale_parameter=args.eta_scale, max_iterations=args.iterations
    )
    
    aco.antColonyOptimization()
    num_cluster, membership = aco.get_membership(aco.best_solution)

    if args.no_save:
        pass
    else:
        name = 'ACO_{}_{}.npy'.format(dataset.stem, np.round(aco.best_cost, 4))
        np.save(ans_dir / Path(name), membership)

    

if __name__ == '__main__':
    args = parser_ACO()
    main(args)