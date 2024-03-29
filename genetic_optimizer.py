import pygad
import numpy as np
from qsm_num_ana import cost, kinematics
import time

# kinematics()

#global variables
function_inputs = [0.225, 1.58,  1.92, -1.55, 1, 1, 1, 1]
function_K = 1

#fitness function 
def fitness(genetic_algorithm_object, solution, solution_index):
    K = cost(solution, numerical=True, nb=100)
    fitness = 1/np.abs(K - function_K)
    return fitness

genetic_algorithm_object = pygad.GA(num_generations=600, num_parents_mating=4, fitness_func=fitness, sol_per_pop=8, num_genes=len(function_inputs), 
                                    mutation_num_genes=5, mutation_type='random', mutation_percent_genes=10)

start = time.time()
genetic_algorithm_object.run()
print('Completed in:', round(time.time() - start, 4), 'seconds')

result = genetic_algorithm_object.best_solution()
print('solution:', result[0])
print('fitness:', result[1])
print('solution index:', result[2])

cost(result[0], numerical=True, show_plots=True)
