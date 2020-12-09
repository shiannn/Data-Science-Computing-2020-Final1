import copy
import numpy as np
import random

class SimulatedAnnealing():
    def __init__(self, graph):
        self.graph = graph
        #self.pcur = np.random.randint(len(graph.nodes), size=len(graph.nodes))
        self.pcur = Solution(graph)
        #print(self.pcur.x)
        #print(self.pcur.y)
        self.pnew = copy.copy(self.pcur)
        self.pb = None
        self.max_iterations = 10
        self.temperature = np.inf
        self.min_temperature = 0.5
        self.Tstart = self.init_temperature()
        self.epsilon = 0.025

    
    def init_temperature(self):
        return 300

    def simulated_annealing(self):
        self.pcur.y = self.f_objective_function(self.pcur.x)
        print(self.pcur.y)
        self.pb = copy.copy(self.pcur)
        iteration = 0
        while(not self.c_termination(iteration, self.temperature)):
            self.pnew.x = self.mutation(self.pcur.x)
            self.pnew.y = self.f_objective_function(self.pnew.x)
            print(self.pnew.y)
            dE = -(self.pnew.y - self.pcur.y)
            #print(iteration, self.pcur.x, self.pcur.y)
            #print(dE, self.pnew.y, self.pcur.y)
            if dE <= 0:
                self.pcur = copy.copy(self.pnew)
                if self.pcur.y > self.pb.y:
                    self.pb = copy.copy(self.pcur)
            else:
                self.temperature = self.getTemp(iteration)
                if np.random.rand() < np.exp(-dE/self.temperature):
                    self.pcur = copy.copy(self.pnew)

            iteration += 1
        
        print('simulated annealing', self.pb.x, self.pb.y)
        return self.pb
    
    def getTemp(self, iteration):
        decay = (1 - self.epsilon)**iteration
        return (decay* self.Tstart)

    def mutation(self, x):
        components_mut = nx.Graph()
        for node in self.graph.nodes:
            components_mut.add_edge(node, random.choice(list(self.graph.neighbors(node))))
        x_mut = list(nx.connected_components(components_mut))
        
        return x_mut
    
    def c_termination(self, iteration, temperature):
        if iteration >= self.max_iterations:
            return True
        if self.temperature <= self.min_temperature:
            return True

    def f_objective_function(self, x):
        ### x should be a Solution
        score = nx_comm.modularity(self.graph, x)
        
        return score

class Solution():
    def __init__(self, graph):
        self.components = nx.Graph()
        for node in graph.nodes:
            self.components.add_edge(node, random.choice(list(graph.neighbors(node))))
        self.x = list(nx.connected_components(self.components))
        self.y = nx_comm.modularity(graph, self.x)
def main():
    kara_set = mmread(str(karate_dataset))
    graph = nx.from_scipy_sparse_matrix(kara_set)
    #ret = treetraverse(graph)
    #pso = ParticleSwarm(graph = graph, N_swarm_size=5, D_dimension=len(graph.nodes), c1=1, c2=1, max_iterations=10)
    sa = SimulatedAnnealing(graph)
    #global_center, global_best = pso.particleSwarm()
    sa.simulated_annealing()
    

    #score = nx_comm.modularity(A, [{a for a in range(10)}, {a for a in range(10,34)}])
    #print(score)

if __name__ == '__main__':
    import networkx as nx
    from scipy.io import mmread
    import networkx.algorithms.community as nx_comm
    from config import karate_dataset
    main()