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

class GeneticAlgorithm():
    def __init__(self, graph, mtx, popu_size = 10, iterations = 20, mating_pool_size=4, flip_locations=2):
        self.graph = graph
        self.mtx = mtx

        self.population_size = popu_size
        self.max_generations = iterations
        self.mating_pool_size = mating_pool_size
        self.k_cross_point = 2
        self.flip_locations = flip_locations

        self.plot_list = []
        self.best_value = 0
    
    def get_membership(self, locus_based):
        row = np.arange(locus_based.shape[0])
        col = locus_based[row]
        attempt_graph = csr_matrix((np.ones(row.shape), (row, col)), shape=(row.shape[0], col.shape[0]))
        num_cluster, membership = connected_components(attempt_graph)

        return num_cluster, membership

    def get_locus(self, x):
        #chosen_neighbor_ids = np.floor(np.diff(self.mtx.indptr)* x).astype(int)
        chosen_neighbor_ids = x
        #print('chosen_neighbor_ids', chosen_neighbor_ids)
        neighs_pos = chosen_neighbor_ids + self.mtx.indptr[:-1]
        locus_based = self.mtx.indices[neighs_pos]
        #print('locus_based', locus_based)

        return locus_based

    def f_objective_function(self, x):
        scores = []
        for population in x:
            locus_based = self.get_locus(population)
            #print(locus_based)
            num_cluster, membership = self.get_membership(locus_based)
            score = self.graph.modularity(membership)
            scores.append(score)
            #print(score)
        scores = np.array(scores)
        #print(scores)
        
        return scores
    
    def genetic_algorithm(self):
        self.best_value, self.plot_list = 0, []
        best_sol = None
        
        print(np.diff(self.mtx.indptr))
        new_population = np.random.randint(
            low=np.zeros(self.graph.vcount()), high=np.diff(self.mtx.indptr),
            size=(self.population_size, self.graph.vcount())
        )
        print(new_population.shape)

        for generation in range(self.max_generations):
            #print("Generation : ", generation)
            #print(new_population)
            popu_scores = self.f_objective_function(new_population)
            #print(popu_scores)
            """
            ### plot
            if np.min(fitness) == 0:
                to_plot = 0
            else:
                to_plot = self.get_survival_point(new_population)[np.argmin(fitness)]
            """

            if popu_scores.max() > self.best_value:
                best_sol = new_population[np.argmax(popu_scores)]
                
            self.best_value = max(self.best_value, popu_scores.max())
            #self.plot_list.append(self.best_value)
            #print(new_population)
            #print(best_sol, self.best_value)
            
            ### select parent
            select_prob = popu_scores / popu_scores.sum()
            parents_idx = np.random.choice(np.arange(new_population.shape[0]), size=self.mating_pool_size, p = select_prob)
            parents = new_population[parents_idx]
            
            ### crossover
            offsprings = []
            for _ in range(self.population_size - self.mating_pool_size):
                gp_idxs = np.random.randint(0, self.mating_pool_size, 2)
                gp = parents[gp_idxs]
                #gp1, gp2 = random.sample(parents, 2)
                child = self.multi_point_crossover(gp)
                offsprings.append(child)
            offsprings = np.array(offsprings)

            #print(offsprings)
            ### mutation
            mutated_offsprings = self.multi_bit_mutation(offsprings)
            new_population = np.concatenate([parents, mutated_offsprings], axis=0)
            
            print(self.best_value)
        return best_sol, self.best_value
    
    def multi_bit_mutation(self, offsprings):
        #print(offsprings)
        mut = np.random.randint(self.graph.vcount(), size=(offsprings.shape[0], self.flip_locations))
        #print(mut)
        available_neighbors = np.diff(self.mtx.indptr)
        new_id = np.random.randint(low=np.zeros(mut.flatten().shape), high=available_neighbors[mut.flatten()], size=mut.flatten().shape)
        
        #new_id = new_id.reshape(-1, self.flip_locations)
        #new_neighbors = self.mtx.indices[new_id]
        mut_rows = np.arange(mut.shape[0]).repeat(self.flip_locations)
        mut_cols = mut.flatten()
        #print(mut_rows)
        #print(mut_cols)
        offsprings[mut_rows, mut_cols] = new_id
        #print(offsprings)
        """
        for f in range(self.flip_locations):
            #print(offsprings[np.arange(mut.shape[0]), mut[:,f]])
            available_neighbors = np.diff(self.mtx.indptr)
            new_id = np.random.randint(low=np.zeros(mut[:,f].shape), high=available_neighbors[mut[:,f]], size=mut[:,f].shape)
            #new_neighbors = self.mtx.indices[new_id]
            offsprings[np.arange(mut.shape[0]), mut[:,f]] = new_id
        """
        #print(offsprings)
        
        return offsprings

    def multi_point_crossover(self, gp):
        ### Uniform crossover
        num_points = gp.shape[1]
        logits = np.random.rand(num_points)
        #print(logits)
        #print(gp)
        g = np.where(logits>=0.5, gp[1], gp[0])
        #print(g)
        return g

def main():
    #dataset = karate_dataset
    dataset = coauthors_dataset
    print('read mtx file')
    mtx = mmread(str(dataset)).tocsr()
    print('to igraph')
    srcs, tgts = mtx.nonzero()
    graph = Graph(list(zip(srcs.tolist(), tgts.tolist())))
    print('start')
    ga = GeneticAlgorithm(graph, mtx, popu_size = 30, iterations = 20, mating_pool_size=10, flip_locations=100)
    best_sol, best_value = ga.genetic_algorithm()
    """
    global_center, global_best = pso.particleSwarm()
    best_locus = pso.get_locus(global_center)
    num_cluster, membership = pso.get_membership(best_locus)

    name = 'PSO_{}_{}.npy'.format(dataset.stem, np.round(global_best, 4))
    np.save(ans_dir / Path(name), membership)
    """

if __name__ == '__main__':
    main()