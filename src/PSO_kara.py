import numpy as np

class ParticleSwarm():
    def __init__(self, graph, lower_bound=-4, upper_bound=4, max_iterations=20, N_swarm_size=4, D_dimension=34, 
        c1=1, c2=1, w=1, swarm_position=None, swarm_velocity=None):

        self.graph = graph
        self.swarm_size = N_swarm_size
        self.dimension = D_dimension

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if swarm_position is None:
            self.swarm_position = self.lower_bound + (self.upper_bound - self.lower_bound)* np.random.rand(
                self.swarm_size, self.dimension
            )
        else:
            self.swarm_position = position

        self.swarm_velocity = np.zeros((self.swarm_size, self.dimension))

        self.local_best = ((-np.inf)* np.ones(self.swarm_size)).squeeze()
        self.global_best = -np.inf
        self.local_center = ((-np.inf)* np.ones((self.swarm_size, self.dimension))).squeeze()
        self.global_center = -np.inf

        self.c1 = c1
        self.c2 = c2
        self.w = w

        self.max_iterations = max_iterations
        #print(self.graph.subgraph([1,20,33]).edges)
    
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

        #print('ans', self.global_center, self.global_best)
        return self.global_center, self.global_best

    def converge(self, position):
        score = self.f_objective_function(position)
        
        #return (values == values[0]).all()
        return (score > 0.8).any()
        
    def f_objective_function(self, x):
        y = np.where(x<=0, np.ones(x.shape), 2* np.ones(x.shape))
        #y = np.where(x<=0)
        scores = []
        for particle_idx in range(self.swarm_size):
            community1, = np.where(y[particle_idx]==1)
            community2, = np.where(y[particle_idx]==2)
            #score = nx_comm.modularity(self.graph, [{a for a in range(10)}, {a for a in range(10,34)}])
            score = nx_comm.modularity(self.graph, [community1, community2])
            #print(score)
            scores.append(score)
        scores = np.array(scores)
        #print(scores)
        
        return scores

def main():
    kara_set = mmread(str(karate_dataset))
    graph = nx.from_scipy_sparse_matrix(kara_set)
    pso = ParticleSwarm(graph, N_swarm_size=5, c1=1, c2=1, max_iterations=10)
    global_center, global_best = pso.particleSwarm()
    print(global_best)
    print(np.where(global_center>0))

    #score = nx_comm.modularity(A, [{a for a in range(10)}, {a for a in range(10,34)}])
    #print(score)

if __name__ == '__main__':
    import networkx as nx
    from scipy.io import mmread
    import networkx.algorithms.community as nx_comm
    from config import karate_dataset
    main()
    """
    pso = ParticleSwarm(lower_bound=-4, upper_bound=4, N_swarm_size=4, D_dimension=1, c1=1, c2=1, w=1)
    #print(pso.swarm_position)
    #print(pso.swarm_velocity)
    #print(pso.f_objective_function(pso.swarm_position))
    pso.particleSwarm()
    """