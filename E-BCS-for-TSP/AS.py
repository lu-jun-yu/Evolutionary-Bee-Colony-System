import numpy as np
import time


class AntColonyOptimization:

    def __init__(self, distances, num_ants, num_iterations, decay, alpha, beta, q0):
        self.distances = distances  # 距离矩阵
        self.num_ants = num_ants    # 蚂蚁的数量
        self.num_iterations = num_iterations  # 迭代次数
        self.decay = decay          # 信息素的蒸发系数
        self.alpha = alpha          # 信息素重要程度的参数
        self.beta = beta            # 启发式信息的重要程度的参数

        self.num_nodes = len(distances)  # 节点的数量
        self.pheromones = np.ones((self.num_nodes, self.num_nodes)) \
                          / np.mean(distances[distances != 0]) / self.num_nodes  # 初始化信息素矩阵
        self.shortest_path = {'shortest_length': np.inf, 'shortest_path': []}  # 初始化最短路径记录
        self.length_history = []  # 记录每次迭代的最短路径长度

    def run(self):
        start = time.time()

        for iteration in range(self.num_iterations):
            paths = []  # 存储当前迭代所有蚂蚁的路径
            path_lengths = []  # 存储当前迭代所有蚂蚁的路径长度

            for ant in range(self.num_ants):
                start_node = np.random.randint(0, self.num_nodes)  # 随机选择起始节点
                current_node = start_node
                path = [start_node]  # 初始化当前蚂蚁的路径
                path_length = 0  # 初始化当前蚂蚁的路径长度

                while True:
                    next_node = self.select_next_node(current_node, path)
                    path.append(next_node)
                    path_length += self.distances[current_node][next_node]
                    if next_node == start_node:
                        break
                    current_node = next_node

                paths.append(path)
                path_lengths.append(path_length)

                if path_length < self.shortest_path['shortest_length']:
                    self.shortest_path['shortest_length'] = path_length
                    self.shortest_path['shortest_path'] = path

                # 记录时间
                end = time.time()
                # # 设置早停
                # if end - start > self.max_run_time:
                #     break

            self.length_history.append(self.shortest_path['shortest_length'])

            # # 设置早停
            # if end - start > self.max_run_time:
            #     break

            # 更新信息素
            self.update_pheromones(paths, path_lengths)

        return self.shortest_path

    def select_next_node(self, current_node, path):
        probabilities = []

        for next_node in range(self.num_nodes):
            if next_node not in path:
                tau_eta = (self.pheromones[current_node][next_node] ** self.alpha) * \
                          ((1 / self.distances[current_node][next_node]) ** self.beta)
                probabilities.append(tau_eta)
            else:
                probabilities.append(0)  # 不可去的节点概率为0

        sum_probabilities = sum(probabilities)
        if sum_probabilities == 0:
            return path[0]

        probabilities = np.array(probabilities) / sum_probabilities
        next_node = np.random.choice(range(self.num_nodes), p=probabilities)
        return next_node

    def update_pheromones(self, paths, path_lengths):
        # 应用信息素衰减
        self.pheromones *= (1 - self.decay)

        for path, path_length in zip(paths, path_lengths):
            for i in range(len(path) - 1):
                self.pheromones[path[i]][path[i + 1]] += 1 / path_length  # 增加信息素
