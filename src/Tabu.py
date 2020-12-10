import copy
import numpy as np
import random
import pickle

class TabuSearch():
    def __init__(self, graph):
        self.graph = graph

        self.pcur = Solution(graph)
        self.pbest = copy.copy(self.pcur)
        #self.pnew = copy.copy(self.pcur)
        #self.ptest = copy.copy(self.pcur)
        self.pnew = None
        self.ptest = None
        self.tabu = []
        self.max_tabu_len = 15
        self.max_iterations = 10

        #self.moves = [
        #    (a, b) for a in range(1, self.distance_matrix.shape[0]) for b in range(a+1, self.distance_matrix.shape[0])
        #]
        #print(self.pcur.x)
        #print(self.pcur.disjointSet.edges)
        self.moves = [
            (a, b) for a in range(len(self.graph.nodes)) for b in self.graph.neighbors(a)
        ]

        self.plot_list = []
    
    def tabu_search(self):
        #self.pbest.x = self.initial_x()
        #self.pbest.x = self.initial_x_with_randomWalk()
        #self.pbest.y = self.f_objective_function(self.pbest.x)
        #self.pcur = copy.copy(self.pbest)
        self.tabu = []
        self.plot_list = []

        iteration = 0
        while(not self.c_termination(iteration)):#or self.pcur
            iteration += 1
            self.pnew = None
            moveb = None
            #print(self.pcur.x, self.pcur.y)
            #for move in self.moves:
            for _ in range(10):
                sample_moves = random.sample(self.moves, 3)
                #print(sample_moves)
                #exit(0)
                for move in sample_moves:
                    self.ptest = self.mutation(self.pcur, move)
                    #print(self.ptest.x)
                    self.ptest.y = self.f_objective_function(self.ptest)
                    COND = ((move not in self.tabu) and (((self.pnew is None) or (self.ptest.y > self.pnew.y)) or self.ptest.y > self.pbest.y))
                    if iteration == 1:
                        print(self.ptest.y)
                    if COND:
                        self.pnew = copy.copy(self.ptest)
                        moveb = move
                    #print('self.pnew.y', self.pnew.y)
            #if iteration == 3:
            #    exit(0)
            self.pcur = copy.copy(self.pnew)
            if self.pcur:
                if self.pcur.y >= self.pbest.y:
                    self.pbest = copy.copy(self.pcur)
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
        ### substitution
        ### change carry
        #print('before pcur.x', pcur.x)
        #print(pcur.disjointSet.edges)
        #print(move)
        node, new_neighbor = move
        old_neighbor = pcur.x[node]
        ### Modify hand list x
        pcur.x[node] = new_neighbor
        ### Modify disjointSet
        pcur.disjointSet.add_edge(node, new_neighbor)
        ### if the old_neighbor and you don't carry, remove the edge
        if pcur.x[old_neighbor] != node and pcur.x[node] != old_neighbor:
            pcur.disjointSet.remove_edge(node, old_neighbor)
        #print('after pcur.x', pcur.x)
        #print(pcur.disjointSet.edges)
        return pcur

    def f_objective_function(self, pcur):
        ### pcur should be a Solution
        components = list(nx.connected_components(pcur.disjointSet))
        score = nx_comm.modularity(self.graph, components)
        return score

    def c_termination(self, iteration):
        return (iteration >= self.max_iterations)

class Solution():
    def __init__(self, graph):
        #self.components = nx.Graph()
        self.disjointSet = nx.Graph()
        self.x = (-1)* np.ones(len(graph.nodes), dtype=int)
        for node in graph.nodes:
            neigh = random.choice(list(graph.neighbors(node)))
            #self.components.add_edge(node, neigh))
            self.disjointSet.add_edge(node, neigh)
            self.x[node] = neigh
        #self.x = list(nx.connected_components(self.components))
        components = list(nx.connected_components(self.disjointSet))
        self.y = nx_comm.modularity(graph, components)

def main():
    kara_set = mmread(str(karate_dataset))
    graph = nx.from_scipy_sparse_matrix(kara_set)
    tabuSearch = TabuSearch(graph)
    #print(tabuSearch.moves)
    
    tabuSearch.tabu_search()

if __name__ == '__main__':
    import networkx as nx
    from scipy.io import mmread
    import networkx.algorithms.community as nx_comm
    from config import karate_dataset
    main()