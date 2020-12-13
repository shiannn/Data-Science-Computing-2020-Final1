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
            ants = [Ant(self) for i in range(self.N_particle_size)]
            for ant in ants:
                for i in range(self.distance_matrix.shape[0] - 1):
                    ant.move()
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
        

    def f_objective_function(self, x_tuple):
        ### x should be a tuple
        total_distance = 0
        num_city = len(x_tuple)
        for idx in range(num_city):
            total_distance += self.distance_matrix[x_tuple[idx], x_tuple[(idx+1)%num_city]]
        return total_distance
            

class Ant():
    def __init__(self, aco):
        self.aco = aco
        self.total_value = 0

        self.delta_pheromone = []

        #self.start_city = np.random.randint(aco.distance_matrix.shape[0])
        self.start_city = 0
        self.tabu = []
        self.tabu.append(self.start_city)
        self.neighborhood = [i for i in range(aco.distance_matrix.shape[0])]
        self.neighborhood.remove(self.start_city)
        self.current_state = self.start_city

    def move(self):
        ### calculate probability from neighborhood
        denominator = sum(
            [self.aco.pheromone_matrix[self.current_state, neighbor]**self.aco.alpha
            for neighbor in self.neighborhood]
        )
        #print(denominator)
        possibilities = [
            (self.aco.pheromone_matrix[self.current_state, neighbor]**self.aco.alpha)/denominator
            for neighbor in self.neighborhood
        ]
        #print(self.aco.pheromone_matrix)
        #print(possibilities)
        next_city = np.random.choice(self.neighborhood, 1, p=possibilities).item()
        #print(next_city)
        self.total_value += self.aco.distance_matrix[self.current_state, next_city]
        self.neighborhood.remove(next_city)
        self.tabu.append(next_city)
        self.current_state = next_city

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
    

    #score = nx_comm.modularity(A, [{a for a in range(10)}, {a for a in range(10,34)}])
    #print(score)

if __name__ == '__main__':
    main()