# 导入所需的库
import numpy as np
import time


# 定义蚁群系统算法类
class AntColonySystem:

    # 类的初始化函数
    def __init__(self, distances, num_ants, num_iterations, global_decay, local_decay, alpha, beta, q0):
        self.distances = distances
        self.num_ants = num_ants  # 蚂蚁的数量
        self.num_iterations = num_iterations  # 迭代次数

        self.global_decay = global_decay
        self.local_decay = local_decay  # 信息素的衰减系数
        self.alpha = alpha  # 信息素重要程度的参数
        self.beta = beta  # 兴趣值重要程度的参数
        self.q0 = q0  # 探索因子，用于控制探索的概率

        self.num_nodes = len(distances)  # 景点的数量
        self.pheromones = np.ones((self.num_nodes, self.num_nodes)) \
                          / np.mean(distances[distances != 0]) / self.num_nodes  # 初始化信息素矩阵
        self.shortest_path = {'shortest_length': np.inf, 'shortest_path': []}  # 最佳路径
        self.length_history = []  # 记录每次迭代的最佳兴趣值，用于后续的可视化

    # 定义蚁群系统的运行函数
    def run(self):
        start = time.time()

        # 开始迭代
        for iteration in range(self.num_iterations):
            paths = []  # 存储所有蚂蚁的路径
            path_lengths = []  # 存储所有路径的兴趣值总和

            # 遍历所有的蚂蚁
            for ant in range(self.num_ants):
                start_node = np.random.randint(0, self.num_nodes)
                current_node = start_node
                path = [start_node]  # 初始化当前蚂蚁的路径
                path_length = 0  # 初始化当前路径的兴趣值总和

                # 循环，直至时间耗尽或者完成旅行
                while True:
                    # 选择下一个景点
                    next_node = self.select_next_node(current_node, path)
                    # 更新路径和时间
                    path.append(next_node)
                    # 更新兴趣值
                    path_length += self.distances[current_node][next_node]
                    if next_node == start_node:
                        break
                    # 更新当前景点
                    current_node = next_node

                # 如果完成了一条完整的路径，则记录该路径及其兴趣值
                paths.append(path)
                path_lengths.append(path_length)

                # 更新全局最佳路径和兴趣值
                if path_length < self.shortest_path['shortest_length']:
                    self.shortest_path['shortest_length'] = path_length
                    self.shortest_path['shortest_path'] = path

                # 记录时间
                end = time.time()
                # # 设置早停
                # if end - start > self.max_run_time:
                #     break

            # 记录这次迭代的最佳兴趣值，用于后续可视化
            self.length_history.append(self.shortest_path['shortest_length'])

            # # 设置早停
            # if end - start > self.max_run_time:
            #     break

            # 更新信息素
            self.global_pheromones_update()

        # 返回最佳路径和兴趣值
        return self.shortest_path

    # 定义选择下一个景点的函数
    def select_next_node(self, current_node, path):
        probabilities = []  # 存储转移到每个景点的概率

        # 计算到每个景点的概率
        for next_node in range(self.num_nodes):
            # 计算到下一个景点将要花费的时间
            # 如果下一个景点不在当前路径中，并且时间允许，则计算概率
            if next_node not in path:
                # 计算景点的信息素和兴趣值的组合影响力
                tau_eta = (self.pheromones[current_node][next_node] **
                           self.alpha) * \
                          ((1 / self.distances[current_node][next_node]) **
                           self.beta)  ### 这一行有修改
                probabilities.append(tau_eta)
            else:
                probabilities.append(0)  # 如果景点不可达或已经访问过，则概率为0

        sum_probabilities = sum(probabilities)
        if sum_probabilities == 0:
            next_node = path[0]
            # 执行局部信息素更新
            self.local_pheromone_update(current_node, next_node)
            return next_node

        # 归一化概率
        probabilities = np.array(probabilities)
        probabilities /= sum_probabilities

        # 根据探索因子q0决定是贪婪选择还是随机选择下一个景点
        if np.random.rand() < self.q0:
            # 开发：选择概率最高的景点
            next_node = np.argmax(probabilities)
        else:
            # 探索：根据概率分布随机选择景点
            next_node = np.random.choice(range(self.num_nodes), p=probabilities)

        # 执行局部信息素更新
        self.local_pheromone_update(current_node, next_node)

        return next_node

    # 定义全局信息素更新函数
    def global_pheromones_update(self):
        # 应用信息素衰减
        self.pheromones *= (1 - self.global_decay)

        # 遍历最优路径中的每段旅程
        for j in range(len(self.shortest_path['shortest_path']) - 1):
            # 增加新的信息素
            self.pheromones[self.shortest_path['shortest_path'][j]][self.shortest_path['shortest_path'][j + 1]] += \
                self.global_decay * (1 / self.shortest_path['shortest_length'])

    # 定义局部信息素更新函数
    def local_pheromone_update(self, current_node, next_node):
        # 局部更新规则
        self.pheromones[current_node][next_node] *= (1 - self.local_decay)
        self.pheromones[current_node][next_node] += self.local_decay * (
                1 / self.distances[current_node][next_node] / self.num_nodes)
