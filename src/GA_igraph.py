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

from argument import parser_GA

class GeneticAlgorithm():
    def __init__(self, graph, mtx, popu_size = 10, iterations = 20, mating_pool_size=4, k_cross_point=2, flip_locations=2):
        self.graph = graph
        self.mtx = mtx

        self.population_size = popu_size
        self.max_generations = iterations
        self.mating_pool_size = mating_pool_size
        self.k_cross_point = k_cross_point
        self.cross_idx = np.random.randint(graph.vcount(), size=self.k_cross_point)

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
            #select_prob = popu_scores / popu_scores.sum()
            select_prob = np.exp(popu_scores) / np.exp(popu_scores).sum()
            parents_idx = np.random.choice(np.arange(new_population.shape[0]), size=self.mating_pool_size, p = select_prob)
            parents = new_population[parents_idx]
            
            ### crossover
            #offsprings = []
            offsprings_num = self.population_size - self.mating_pool_size
            gp_idxs = np.random.randint(self.mating_pool_size, size=(offsprings_num, 2))
            gp1 = parents[gp_idxs[:,0]]
            gp2 = parents[gp_idxs[:,1]]
            #offsprings = self.uniform_crossover(gp1, gp2)
            offsprings = self.multi_point_crossover(gp1, gp2)

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
        
        return offsprings

    def multi_point_crossover(self, gp1, gp2):
        ### Multi point crossover
        #cross_pt = np.random.randint(2, size=gp1.shape[1])
        cross_idx = np.random.randint(self.graph.vcount(), size=self.k_cross_point)
        cross_helper = np.zeros(gp1.shape[1])
        cross_helper[cross_idx] = 1
        cross_helper = np.cumsum(cross_helper)
        cross_helper[cross_helper%2==0] = 0
        cross_helper[cross_helper%2==1] = 1
        #print(cross_helper)
        g = np.where(cross_helper==0, gp1, gp2)

        return g

    def uniform_crossover(self, gp1, gp2):
        ### Uniform crossover
        g = np.zeros(gp1.shape)
        logits = np.random.rand(g.shape[0], g.shape[1])
        #print(logits)
        g = np.where(logits>=0.5, gp2, gp1)
        #print(g.shape)
        
        return g

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
    ga = GeneticAlgorithm(
        graph, mtx, 
        popu_size = args.popu_size, iterations = args.iterations, mating_pool_size=args.mate_pool_size,
        k_cross_point=args.k_cross_points, flip_locations=args.flip_locations
    )
    best_sol, best_value = ga.genetic_algorithm()
    
    best_locus = ga.get_locus(best_sol)
    num_cluster, membership = ga.get_membership(best_locus)

    if args.no_save:
        pass
    else:
        name = 'GA_{}_{}.npy'.format(dataset.stem, np.round(best_value, 4))
        np.save(ans_dir / Path(name), membership)
    

if __name__ == '__main__':
    args = parser_GA()
    main(args)