import numpy as np

class ParticleSwarm():
    def __init__(self, lower_bound, upper_bound, N_swarm_size, D_dimension, 
        c1, c2, w, swarm_position=None, swarm_velocity=None):

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

        self.local_best = ((-np.inf)* np.ones((self.swarm_size, self.dimension))).squeeze()
        self.global_best = -np.inf
        self.local_center = ((-np.inf)* np.ones((self.swarm_size, self.dimension))).squeeze()
        self.global_center = -np.inf

        self.c1 = c1
        self.c2 = c2
        self.w = w

        self.max_iterations = 20
    
    def particleSwarm(self):
        for iteration in range(self.max_iterations):
            if self.converge(self.swarm_position):
                break
            print(iteration)
            print(self.swarm_position)
            encountered_by_particles = self.f_objective_function(self.swarm_position)
            print(encountered_by_particles)

            self.local_center = np.where(encountered_by_particles > self.local_best, self.swarm_position, self.local_center)
            self.local_best = np.where(
                encountered_by_particles > self.local_best, encountered_by_particles, self.local_best
            )
            self.global_center = self.swarm_position[np.argmax(encountered_by_particles)] if np.max(encountered_by_particles) > self.global_best else self.global_center
            self.global_best = max(self.global_best, np.max(encountered_by_particles))

            local_effect = np.diag(self.local_center - self.swarm_position).reshape(-1,1)
            global_effect = self.global_center - self.swarm_position
            
            self.swarm_velocity = (self.w* self.swarm_velocity +
                self.c1* np.random.rand()* local_effect +
                self.c2* np.random.rand()* global_effect
            )

            self.swarm_position = self.swarm_position + self.swarm_velocity

        print('ans', self.global_center, self.global_best)

    def converge(self, position):
        values = self.f_objective_function(position)
        
        return (values == values[0]).all()
        
    def f_objective_function(self, x):
        f_ret = -x**5 + 5*x**3 + 20*x - 5
        f_ret = f_ret.squeeze()
        f_ret = np.where(np.logical_or(x>self.upper_bound, x < self.lower_bound).squeeze(), np.nan, f_ret)
        
        return f_ret

if __name__ == '__main__':
    pso = ParticleSwarm(lower_bound=-4, upper_bound=4, N_swarm_size=4, D_dimension=1, c1=1, c2=1, w=1)
    #print(pso.swarm_position)
    #print(pso.swarm_velocity)
    #print(pso.f_objective_function(pso.swarm_position))
    pso.particleSwarm()