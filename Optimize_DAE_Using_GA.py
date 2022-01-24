# Optimization the denoising autoencoder by genetic algorithm for anomaly detection (DAEGA)

import numpy as np

input_size = 5
hidden_size = 3

# 迭代次数
num_generations = 100
# 染色体长度
chr_size = 2*input_size + input_size*hidden_size*2 + hidden_size
# 每个群落中染色体的数量
chr_per_pop = 8 
# 定义群落规模，(染色体数量，染色体长度).
population_size = (chr_per_pop, chr_size)
# 初始化群落.
new_population = np.random.uniform(low=-1.0, high=1.0, size=population_size)
# 群落中精英染色体数量
num_parents_mating = 4 
# 交配得到的下一代基因来自父染色体的概率
p = 0.5
# 每条染色变异基因的最大数量（不能大于chr_size）
num_mutation = int(chr_size * 0.1)


# 降噪自编码器
# 输入：染色体和输入x
# 输出：x重构后的数据y
def DAE(chromosome, input):

    noise = chromosome[0 : input_size]
    inputAddNoise = input + noise

    w_index1 = input_size + input_size*hidden_size
    w_index2 = w_index1 + hidden_size*input_size
    b_index1 = w_index2 + hidden_size
    b_index2 = b_index1 + input_size

    w_in_to_hidden = chromosome[input_size : w_index1].reshape(hidden_size, input_size)
    w_hidden_to_out = chromosome[w_index1 : w_index2].reshape(input_size, hidden_size)
    b_in_to_hidden = chromosome[w_index2 : b_index1]
    b_hidden_to_out = chromosome[b_index1 : b_index2]

    # print("inputAddNoise: ", inputAddNoise)
    # print("w_in_to_hidden: ", w_in_to_hidden)
    # print("w_hidden_to_out: ", w_hidden_to_out)
    # print("b_in_to_hidden: ", b_in_to_hidden)
    # print("b_hidden_to_out: ", b_hidden_to_out)

    hidden = relu(np.dot(w_in_to_hidden, inputAddNoise) + b_in_to_hidden)
    # print("w_in_to_hidden*inputAddNoise: ", np.dot(w_in_to_hidden, inputAddNoise))
    # print("hidden: ",hidden)
    output = relu(np.dot(w_hidden_to_out, hidden) + b_hidden_to_out)

    return output

def relu(array):
    return np.maximum(array, 0)

# 计算适应度，即输入x与x重构后的y直接的距离，这个值越小表示染色体越好
def col_pop_fitness(x, pop):
    fitness = []
    for i in range(chr_per_pop):
        chr = pop[i, :] # 染色体
        y = DAE(chr, x)
        dist = np.sqrt(np.sum(np.square(x - y)))
        fitness.append(dist)
    fitness = np.array(fitness)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best chromosome in the current generation as parents for producting the offspring of the next generation.
    parents = np.empty((num_parents, chr_size))
    for parent_num in range(num_parents):
        min_fitness_idx = np.where(fitness == np.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent_num, :] = pop[min_fitness_idx, :]
        fitness[min_fitness_idx] = float('inf')
    return parents

# 给精英染色体和交配得到的子代规模(染色体总数量-精英染色体数量, 染色体长度)
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = np.random.randint(0, parents.shape[0])
        # Index of the second parent to mate
        parent2_idx = np.random.randint(0, parents.shape[0])

        is_from_a = np.random.random(chr_size) < p
        offspring[k, :] = np.where(is_from_a, parents[parent1_idx, :], parents[parent2_idx, :])

    return offspring

def mutation(offspring_crossover):
    # Mutation changes some gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(-1.0, 1.0, num_mutation)
        offspring_crossover[idx, np.random.randint(0, chr_size, num_mutation)] = random_value
    return offspring_crossover

def train():

    for generation in range(num_generations):

        x = [0.2,0.4,0.9,0.1,0.5]

        # 计算适应度
        fitness = col_pop_fitness(x, new_population)

        # 在群落中选择适应度最小的染色体做为精英.
        parents = select_mating_pool(new_population, fitness, num_parents_mating)

        # 通过交配生成下一代.
        offspring_crossover = crossover(parents, offspring_size=(population_size[0]-parents.shape[0], chr_size))

        # 基因变异.
        offspring_mutation = mutation(offspring_crossover)

        # 得到下一代.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        # The best result in the current iteration.
        if generation%20 == 0: 
            print("Generation : ", generation)
            fitness = col_pop_fitness(x, new_population)
            print("Best result : ", np.min(fitness))

    # Getting the best solution after iterating finishing all generation
    # At first, the fitness is calculated for each solution in the final generation
    fitness = col_pop_fitness(x, new_population)

    # Then return the index of the solution corresponding the best fitness
    best_match_idx = np.where(fitness == np.min(fitness))[0][0]
    best_chromosome = new_population[best_match_idx, :]
    # print("Best_match_idx : ", best_match_idx)
    print("Best solution : ", best_chromosome)
    print("Best solution fitness : ", fitness[best_match_idx])

    return best_chromosome


if __name__ == '__main__':
    best_chromosome = train()
    