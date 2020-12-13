import copy
import random
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import igraph
from igraph import Graph, VertexClustering
from config import karate_dataset, coauthors_dataset

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
        self.pheromone_array = 1 / np.repeat(self.num_available_neighbors, self.num_available_neighbors)
        print(self.pheromone_array.shape)
        self.prob_array = self.pheromone2prob()
        print(self.prob_array.shape)

        self.best_cost = None
        self.best_solution = None
        
        self.plot_List = []
    
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
        prob = self.pheromone_array / rep_total_phromone
        
        return prob

    def antColonyOptimization(self):
        for iteration in range(self.max_iterations):
            ants = [Ant(self) for _ in range(self.N_particle_size)]
            for ant in ants:
                #for i in range(self.distance_matrix.shape[0] - 1):
                ant.move()
                exit(0)
                #print(ant.tabu)
                #ant.total_value += self.distance_matrix[ant.tabu[0], ant.tabu[-1]]
                #print(self.f_objective_function(ant.tabu))
                if self.best_cost is None or self.f_objective_function(ant.tabu) < self.best_cost:
                    self.best_cost = self.f_objective_function(ant.tabu)
                    self.best_solution = ant.tabu.copy()

            print(self.best_solution)
            print(self.f_objective_function(self.best_solution))
            #print(np.round(self.pheromone_matrix, 2))

            self.plot_List.append(self.best_cost)

            ### the ant finish constructing its solution
            ### add pheromone on its trail
            self.update_pheromone(ants)
        
    def update_pheromone(self, ants):
        #print([ant.total_value for ant in ants])
        pheromone_worst = max([ant.total_value for ant in ants])
        pheromone_best = min([ant.total_value for ant in ants])
        #print(pheromone_best, pheromone_worst)
        sum_delta_pheromone = sum([
            self.eta_scale_parameter* (pheromone_best / pheromone_worst)
            #self.eta_scale_parameter* (1 - pheromone_best / pheromone_worst)
            for ant in ants if ant.total_value <= pheromone_best
        ])
        #print(sum_delta_pheromone)
        ### evaporate
        self.pheromone_matrix *= (1 - self.p_evaporates)
        ### best ant / worst ant and add on edges of best solution
        for ant in ants:
            if ant.total_value <= pheromone_best:
                ### add pheromone on its trail if it is the best
                print('ant')
                for st in range(len(ant.tabu) - 1):
                    self.pheromone_matrix[ant.tabu[st], ant.tabu[st+1]] += sum_delta_pheromone
                    self.pheromone_matrix[ant.tabu[st+1], ant.tabu[st]] += sum_delta_pheromone
        

    def f_objective_function(self, x):
        ### x should be a locus_based vector (num_vertices) with value as chosen neighbor
        row = np.arange(x.shape[0])
        col = x[row]
        attempt_graph = csr_matrix((np.ones(row.shape), (row, col)), shape=(row.shape[0], col.shape[0]))
        num_cluster, membership = connected_components(attempt_graph)
        score = self.graph.modularity(membership)
        
        return score
            

class Ant():
    def __init__(self, aco):
        self.aco = aco
        self.total_value = 0

        self.delta_pheromone = []

        #self.start_city = np.random.randint(aco.distance_matrix.shape[0])
        self.start_city = 0
        self.tabu = []
        self.tabu.append(self.start_city)
        self.current_state = self.start_city

    def move(self):
        ### calculate probability from neighborhood
        """
        denominator = sum(
            [self.aco.pheromone_matrix[self.current_state, neighbor]**self.aco.alpha
            for neighbor in self.neighborhood]
        )
        #print(denominator)
        possibilities = [
            (self.aco.pheromone_matrix[self.current_state, neighbor]**self.aco.alpha)/denominator
            for neighbor in self.neighborhood
        ]
        """
        prob_array = self.aco.pheromone2prob()
        new_neighbors = self.prob2neighbor(prob_array)
        #print(self.aco.pheromone_matrix)
        #print(possibilities)
        #next_city = np.random.choice(self.neighborhood, 1, p=possibilities).item()
        #print(next_city)
        score = self.aco.f_objective_function(new_neighbors)
        print(score)
        exit(0)
        self.total_value += self.aco.distance_matrix[self.current_state, next_city]
        #self.neighborhood.remove(next_city)
        #self.tabu.append(next_city)
        self.current_state = new_neighbors
    
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
        neighbors = self.aco.mtx.indices[cum_selected_neighbors_ids]

        return neighbors

def main():
    dataset = karate_dataset
    #dataset = coauthors_dataset
    print('read mtx file')
    mtx = mmread(str(dataset)).tocsr()
    print('to igraph')
    srcs, tgts = mtx.nonzero()
    graph = Graph(list(zip(srcs.tolist(), tgts.tolist())))
    print('start')
    
    aco = AntColonyOptimization(mtx, graph, p_evaporates=0.1, eta_scale_parameter=0.02, max_iterations=100)
    #sa.simulated_annealing()
    aco.antColonyOptimization()
    

    #score = nx_comm.modularity(A, [{a for a in range(10)}, {a for a in range(10,34)}])
    #print(score)

if __name__ == '__main__':
    main()