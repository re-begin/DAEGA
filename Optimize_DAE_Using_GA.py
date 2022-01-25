# Optimization the denoising autoencoder by genetic algorithm for anomaly detection (DAEGA)

from cgi import print_directory
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score,roc_curve

from Dateloader import *

np.random.seed(0)

input_size = 10
hidden_size = 5
num_epoch = 1

# 迭代次数
num_generations = 100
# 染色体长度
chr_size = 2*input_size + input_size*hidden_size*2 + hidden_size
# 每个群落中染色体的数量
chr_per_pop = 100 
# 定义群落规模，(染色体数量，染色体长度).
population_size = (chr_per_pop, chr_size)
# 初始化群落.
new_population = np.random.uniform(low=-1.0, high=1.0, size=population_size)
# 群落中精英染色体数量
num_parents_mating = 30

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

def train(date):

    # batch_avg_fitness_list = []

    for batch in range(num_epoch):

        # batch_fitness = []

        for idx, x in enumerate(date):

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

            fitness = col_pop_fitness(x, new_population)
            # batch_fitness.append(np.min(fitness))

            if idx%5000 == 0: 
                print("Generation : ", idx)
                print("Best result : ", np.min(fitness))

        # Getting the best solution after iterating finishing all generation
        # At first, the fitness is calculated for each solution in the final generation
        
        fitness = col_pop_fitness(x, new_population)
        # batch_avg_fitness = np.mean(batch_fitness)
        # batch_avg_fitness_list.append(batch_avg_fitness)

        # Then return the index of the solution corresponding the best fitness
        best_match_idx = np.where(fitness == np.min(fitness))[0][0]
        best_chromosome = new_population[best_match_idx, :]
        # print("Best_match_idx : ", best_match_idx)
        print("Best solution : ", best_chromosome)
        print("Best solution fitness : ", fitness[best_match_idx])

    # plt.plot(range(len(batch_avg_fitness_list)), batch_avg_fitness_list)
    # plt.show()
    return best_chromosome

def predict(traindate, test_date):

    best_chromosome = train(date=traindate)

    test_date_after = np.array([[]]*input_size).T
    # print(test_date_after.shape)
    for i in test_date:
        new_i = DAE(best_chromosome, i)
        new_i = np.array(new_i).reshape(1,10)
        test_date_after = np.vstack([test_date_after, new_i])   

    print("test_date_after.shape: ", test_date_after.shape)

    return test_date_after

def distance(x, y):
    distance = []
    for i, j in zip(x, y):
        dist = np.sqrt(np.sum(np.square(i - j)))
        distance.append(dist)
    return distance

def datetrans(distance_arr, label_thre, threshold):
    distance_arr = np.array(distance_arr)
    label = np.array([0]*len(distance_arr))
    y = np.array([0]*len(distance_arr))

    label[distance_arr < label_thre] = 1
    y[distance_arr < threshold] = 1
    return label, y

def getmatrix(x, y):
    acc = accuracy_score(x,y)
    f1 = f1_score(x, y)
    pre = precision_score(x, y)
    print("accuracy is: ", acc)
    print("f1 score is: ", f1)
    print("precision is: ", pre)

if __name__ == '__main__':
    # 台区1原始数据-时间窗口split前数据
    testdate_huifu1_1 = redate_local(testdate_1_1, 10, num_user1)
    testdate_huifu1_2 = redate_local(testdate_1_2, 10, num_user1)
    testdate_huifu1_3 = redate_local(testdate_1_3, 10, num_user1)
    testdate_huifu1_4 = redate_local(testdate_1_4, 10, num_user1)
    # 台区2原始数据-时间窗口split前数据
    testdate_huifu2_1 = redate_local(testdate_2_1, 10, num_user2)
    testdate_huifu2_2 = redate_local(testdate_2_2, 10, num_user2)
    testdate_huifu2_3 = redate_local(testdate_2_3, 10, num_user2)
    testdate_huifu2_4 = redate_local(testdate_2_4, 10, num_user2)
    
    # 台区1预测数据-时间窗口split后数据
    test_date_after1_1 = predict(traindate_1_1, testdate_1_1)
    test_date_after1_2 = predict(traindate_1_2, testdate_1_2)
    test_date_after1_3 = predict(traindate_1_3, testdate_1_3)
    test_date_after1_4 = predict(traindate_1_4, testdate_1_4)
    # 台区2预测数据-时间窗口split后数据
    test_date_after2_1 = predict(traindate_2_1, testdate_2_1)
    test_date_after2_2 = predict(traindate_2_2, testdate_2_2)
    test_date_after2_3 = predict(traindate_2_3, testdate_2_3)
    test_date_after2_4 = predict(traindate_2_4, testdate_2_4)
    
    # 台区1预测数据-恢复时间窗口split前格式
    testdate_predict1_1 = redate_local(test_date_after1_1, 10, num_user1)
    testdate_predict1_2 = redate_local(test_date_after1_2, 10, num_user1)
    testdate_predict1_3 = redate_local(test_date_after1_3, 10, num_user1)
    testdate_predict1_4 = redate_local(test_date_after1_4, 10, num_user1)

    # 台区2预测数据-恢复时间窗口split前格式
    testdate_predict2_1 = redate_local(test_date_after2_1, 10, num_user2)
    testdate_predict2_2 = redate_local(test_date_after2_2, 10, num_user2)
    testdate_predict2_3 = redate_local(test_date_after2_3, 10, num_user2)
    testdate_predict2_4 = redate_local(test_date_after2_4, 10, num_user2)

    # 台区1-数据合并还原为未dateprocess后格式（n, 4）
    test1_origin_col1 = testdate_huifu1_1.reshape(-1,1)
    test1_origin_col2 = testdate_huifu1_2.reshape(-1,1)
    test1_origin_col3 = testdate_huifu1_3.reshape(-1,1)
    test1_origin_col4 = testdate_huifu1_4.reshape(-1,1)

    test1_origin = np.hstack([test1_origin_col1, test1_origin_col2, test1_origin_col3, test1_origin_col4])
    print("test1_origin.shape", test1_origin.shape)

    test1_predict_col1 = testdate_predict1_1.reshape(-1,1)
    test1_predict_col2 = testdate_predict1_2.reshape(-1,1)
    test1_predict_col3 = testdate_predict1_3.reshape(-1,1)
    test1_predict_col4 = testdate_predict1_4.reshape(-1,1)

    test1_predict = np.hstack([test1_predict_col1, test1_predict_col2, test1_predict_col3, test1_predict_col4])
    print("test1_predict.shape", test1_predict.shape)

    # 台区2-数据合并还原为未dateprocess后格式（n, 4）
    test2_origin_col1 = testdate_huifu2_1.reshape(-1,1)
    test2_origin_col2 = testdate_huifu2_2.reshape(-1,1)
    test2_origin_col3 = testdate_huifu2_3.reshape(-1,1)
    test2_origin_col4 = testdate_huifu2_4.reshape(-1,1)

    test2_origin = np.hstack([test2_origin_col1, test2_origin_col2, test2_origin_col3, test2_origin_col4])
    print("test2_origin.shape", test2_origin.shape)

    test2_predict_col1 = testdate_predict2_1.reshape(-1,1)
    test2_predict_col2 = testdate_predict2_2.reshape(-1,1)
    test2_predict_col3 = testdate_predict2_3.reshape(-1,1)
    test2_predict_col4 = testdate_predict2_4.reshape(-1,1)

    test2_predict = np.hstack([test2_predict_col1, test2_predict_col2, test2_predict_col3, test2_predict_col4])
    print("test2_predict.shape", test2_predict.shape)

    distance1 = distance(test1_origin, test1_predict)
    distance2 = distance(test2_origin, test2_predict)

    label1, y1 = datetrans(distance1, 1.4, 5)
    label2, y2 = datetrans(distance2, 0.55, 5)
  
    getmatrix(label1, y1)
    getmatrix(label2, y2)  