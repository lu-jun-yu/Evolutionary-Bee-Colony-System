import numpy as np
import random
import time
import scipy.optimize as opt
from collections import deque
import scipy.special as sp


def inverse_gamma(y, a=0, b=20):
    # 定义目标方程：gamma(x) - y
    func = lambda x: x * 2 ** x - 0.002 * (y + 600) ** 2
    # 寻找目标方程的根，即Gamma函数的逆
    root = opt.root_scalar(func, bracket=[a, b], method='brentq')
    return root.root if root.converged else None

class GdA:

    # 类的初始化函数
    def __init__(self, distances):
        self.distances = distances
        self.num_nodes = len(distances)  # 景点的数量
        self.start = np.random.randint(0, self.num_nodes)
        self.shortest_path = {'shortest_length': np.inf, 'shortest_path': []}  # 最佳路径

    # 定义蜂群系统的运行函数
    def run(self):
        # for start in range(self.num_nodes):
        #     path = [start]  # 初始化当前蚂蚁的路径
        #     path_length = self.find_new_path(path, 0)
        #
        #     # 更新全局最佳路径和兴趣值
        #     if path_length < self.shortest_path['shortest_length']:
        #         self.shortest_path['shortest_length'] = path_length
        #         self.shortest_path['shortest_path'] = path

        path = [self.start]  # 初始化当前蚂蚁的路径
        path_length = self.find_new_path(path, 0)

        # 更新全局最佳路径和兴趣值
        if path_length < self.shortest_path['shortest_length']:
            self.shortest_path['shortest_length'] = path_length
            self.shortest_path['shortest_path'] = path

        # 返回最佳路径和兴趣值
        print(self.shortest_path)
        return self.shortest_path

    # 概率性探索函数
    def find_new_path(self, path, path_length):
        probabilities = []  # 存储转移到每个景点的概率

        for next_node in range(self.num_nodes):
            if next_node not in path:
                tau_eta = 1 / (self.distances[path[-1]][next_node] + 1e-8)
                probabilities.append(tau_eta)
            else:
                probabilities.append(0)  # 如果景点不可达或已经访问过，则概率为0

        sum_probabilities = sum(probabilities)

        if sum_probabilities == 0:
            path_length += self.distances[path[-1]][path[0]]
            path.append(path[0])
            return path_length

        next_node = np.argmax(probabilities)

        path_length += self.distances[path[-1]][next_node]
        path.append(next_node)

        return self.find_new_path(path, path_length)


# %%
class EBCS:
    ### start      #####################################
    class individual:

        def __init__(self, alpha, beta, q0, start):
            self.alpha = {'explorer': alpha, 'developer': alpha}
            self.beta = {'explorer': beta, 'developer': beta}
            self.q0 = {'explorer': q0, 'developer': q0}
            self.start = {'explorer': start, 'developer': start}

            self.v = [{'explorer': 0, 'developer': 0}
                , {'explorer': 0, 'developer': 0},
                      {'explorer': 0, 'developer': 0},
                      {'explorer': 0, 'developer': 0}]  # 粒子移动速度
            self.historic_best_parameter = [self.alpha, self.beta, self.q0, start]
            self.historic_best_fitness = -np.inf

    ### end        #####################################

    # 类的初始化函数
    def __init__(self, distances, num_bees, num_iterations, down_bound,
                 alpha, beta, q0, max_run_time):
        self.distances = distances
        self.num_bees = num_bees  # 蚂蚁的数量
        self.num_iterations = num_iterations  # 迭代次数

        self.global_decay = 0.5
        self.local_decay = 0.5
        self.tail_decay = 0.5
        self.down_bound = down_bound
        self.alpha = alpha  # 信息素重要程度的参数
        self.beta = beta  # 兴趣值重要程度的参数
        self.q0 = q0  # 探索因子，用于控制探索的概率

        self.num_nodes = len(distances)  # 景点的数量
        self.max_run_time = max_run_time
        self.sca_mode = 1 if max_run_time / self.num_nodes >= 2e-5 * self.num_nodes ** 2 else 0
        self.num_tail_nodes = inverse_gamma(self.num_nodes)
        self.initial = GdA(distances).run()
        self.best_local_paths = [self.initial]

        self.attraction_enhancement = 0
        self.repulsion_enhancement = 0
        self.repulsion_init = 1  # (0.1, 2)
        self.repulsion_rate = 1  # (1, 10)
        self.attraction_rate = 5  # (1, 10)
        self.attractive_force_field = np.ones((self.num_nodes, self.num_nodes))
        self.repulsive_force_field = np.ones((self.num_nodes, self.num_nodes))

        self.all_paths = set()
        self.shortest_path = {'shortest_length': self.initial["shortest_length"],
                              'shortest_path': self.initial["shortest_path"]}  # 最佳路径
        self.length_history = []  # 记录每次迭代的最佳兴趣值，用于后续的可视化
        self.pheromones = np.ones((self.num_nodes, self.num_nodes)) \
                          / np.mean(distances[distances != 0]) / self.num_nodes  # 初始化信息素矩阵

        ### start：     #####################################
        self.bee_individual = [self.individual(random.uniform(0.1, 2), random.uniform(1, 5),
                                               random.uniform(0.1, 0.9), i)
                               for i in range(self.num_bees)]
        ### end        #####################################

    # 定义蜂群系统的运行函数
    def run(self):
        global end
        start_time = time.time()
        bee = {'explorer': 0, 'developer': 0}
        paths = {'explorer': [], 'developer': []}  # 存储所有蚂蚁的路径
        path_lengths = {'explorer': [], 'developer': []}  # 存储所有路径的兴趣值总和

        # 开始迭代
        for iteration in range(self.num_iterations):
            print("EBCS:" + str(iteration))
            # 探索阶段
            self.repulsion_enhancement = 0
            result = self.search('explorer', bee, paths, path_lengths, iteration, start_time)
            if result is not None:
                return result

            # 记录时间
            end = time.time()

            # 设置早停
            if end - start_time > self.max_run_time:
                break

            # 开发阶段
            self.attraction_enhancement = 0
            result = self.search('developer', bee, paths, path_lengths, iteration, start_time)
            if result is not None:
                return result

            # 记录时间
            end = time.time()
            # 设置早停
            if end - start_time > self.max_run_time:
                break

        return self.shortest_path

    # 搜索函数
    def search(self, mode, bee, paths, path_lengths, iteration, start_time):
        bee_count = 0
        while True:
            while bee[mode] < self.num_bees:
                if mode == 'explorer':
                    repulsion_init = self.repulsion_init
                    repulsion_rate = self.repulsion_rate
                    self.repulsion_enhancement += repulsion_init / ((1 + bee_count) ** repulsion_rate)
                    self.repulsive_force_field_update(bee[mode])

                start = self.bee_individual[bee[mode]].start[mode]
                pre_path = [start]  # 初始化当前蚂蚁的路径
                path = [0]
                accessible_node = set(range(self.num_nodes))
                accessible_node.remove(start)
                path_length = self.find_new_path(pre_path, path, 0, bee[mode], mode, accessible_node)
                if len(path) != self.num_nodes + 1:
                    return self.shortest_path

                # 如果完成了一条完整的路径，则记录该路径及其兴趣值
                paths[mode].append(path)
                path_lengths[mode].append(path_length)

                # 更新全局最佳路径和兴趣值
                if path_length < self.shortest_path['shortest_length']:
                    self.shortest_path['shortest_length'] = path_length
                    self.shortest_path['shortest_path'] = path

                # 记录这次最佳兴趣值，用于后续可视化
                self.length_history.append(self.shortest_path['shortest_length'])

                bee_count += 1

                # 记录时间
                end = time.time()
                # 设置早停
                if end - start_time > self.max_run_time:
                    return None

                if mode == 'explorer':
                    if path_lengths[mode][-1] < self.best_local_paths[0]['shortest_length']:
                        tmp_path = {'shortest_length': path_lengths[mode][-1], 'shortest_path': paths[mode][-1]}
                        # print(tmp_path)
                        self.best_local_paths.append(tmp_path)

                        return None
                if mode == 'developer':
                    length_decline = self.best_local_paths[-1]['shortest_length'] - path_lengths[mode][-1]
                    if length_decline > 0:
                        attraction_rate = self.attraction_rate
                        self.attraction_enhancement += length_decline * attraction_rate / self.num_nodes / \
                                                       self.best_local_paths[-1]['shortest_length']
                        tmp_path = {'shortest_length': path_lengths[mode][-1], 'shortest_path': paths[mode][-1]}
                        self.best_local_paths[-1] = tmp_path
                        self.attractive_force_field_update(bee[mode])
                        bee_count = 0

                bee[mode] += 1

                # 更新信息素
                if bee['explorer'] + bee['developer'] == self.num_bees:
                    index = iteration * 2
                    self.global_decay = (1 - 2 * self.down_bound) / (1 + np.e ** (-index)) + self.down_bound
                    self.local_decay = 1 - self.global_decay
                    self.tail_decay = 1 - self.global_decay
                    self.global_pheromones_update()

                if mode == 'developer' and bee_count == self.num_nodes // 1:
                    return None
                if mode == 'explorer' and bee_count == self.num_nodes // 2:
                    self.best_local_paths.append(self.shortest_path)

                    return None
            else:
                self.pso(mode, path_lengths, iteration)
                paths[mode] = []  # 存储所有蚂蚁的路径
                path_lengths[mode] = []  # 存储所有路径的兴趣值总和
                bee[mode] = 0

    # 概率性探索函数
    def find_new_path(self, pre_path, final_path, path_length, bee, mode, accessible_node):
        path = final_path if pre_path[-1] == 0 else pre_path
        probabilities = {}  # 存储转移到每个景点的概率
        final_path_tuple = tuple(final_path)
        for next_node in accessible_node:
            if self.sca_mode and pre_path[-1] == 0 and final_path_tuple + (next_node,) in self.all_paths:
                continue
            if mode == 'developer':
                force = self.attractive_force_field[path[-1]][next_node]
            else:
                force = self.repulsive_force_field[path[-1]][next_node]
            tau_eta = (self.pheromones[path[-1]][next_node] **
                       self.bee_individual[bee].alpha[mode]) * \
                      ((1 / (self.distances[path[-1]][next_node] + 1e-8)) **
                       self.bee_individual[bee].beta[mode]) * force
            probabilities[next_node] = tau_eta

        # print(probabilities)
        sum_probabilities = sum(probabilities.values())
        if len(probabilities) == 0:
            if len(pre_path) + len(final_path) == self.num_nodes + 1:
                path_length += self.distances[final_path[-1]][pre_path[0]]
                final_path.extend(pre_path)
                final_path_tuple = tuple(final_path[:-2])
                if final_path_tuple not in self.all_paths:
                    if self.sca_mode:
                        self.all_paths.add(final_path_tuple)
                    return path_length
                path_length -= self.distances[final_path[-2]][final_path[-1]]
                pre_path = [final_path.pop()]
                path_length -= self.distances[final_path[-2]][final_path[-1]]
                accessible_node.add(final_path.pop())
                for i in range(1, len(final_path_tuple) - 1):
                    path_length -= self.distances[final_path[-2]][final_path[-1]]
                    accessible_node.add(final_path.pop())
                    if final_path_tuple[:-i] not in self.all_paths:
                        if 0 < len(accessible_node) <= self.num_tail_nodes:
                            return self.find_optimal_path(pre_path, final_path, path_length, bee, mode, accessible_node)
                        else:
                            return self.find_new_path(pre_path, final_path, path_length, bee, mode, accessible_node)
                else:
                    print(1)
                    return path_length
            else:
                final_path_tuple = tuple(final_path)
                if self.sca_mode:
                    self.all_paths.add(final_path_tuple)
                for i in range(1, len(final_path_tuple) - 1):
                    path_length -= self.distances[final_path[-2]][final_path[-1]]
                    accessible_node.add(final_path.pop())
                    if final_path_tuple[:-i] not in self.all_paths:
                        if 0 < len(accessible_node) <= self.num_tail_nodes:
                            return self.find_optimal_path(pre_path, final_path, path_length, bee, mode, accessible_node)
                        else:
                            return self.find_new_path(pre_path, final_path, path_length, bee, mode, accessible_node)
                else:
                    print(2)
                    return path_length

        if sum_probabilities > 0 and not np.isinf(sum_probabilities):
            probabilities = {key: value / sum_probabilities for key, value in probabilities.items()}

            # 根据探索因子q0决定是贪婪选择还是随机选择下一个景点
            if np.random.rand() < self.bee_individual[bee].q0[mode]:
                # 开发：选择概率最高的景点
                next_node = max(probabilities, key=probabilities.get)
            else:
                # 探索：根据概率分布随机选择景点
                next_node = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        elif np.isinf(sum_probabilities):
            for node in probabilities:
                next_node = node
                if np.isinf(probabilities[node]):
                    break
        else:
            next_node = np.random.choice(list(probabilities.keys()))

        path_length += self.distances[path[-1]][next_node]
        path.append(next_node)
        accessible_node.remove(next_node)

        # 执行局部信息素更新
        self.local_pheromone_update(path[-2], path[-1])

        if 0 < len(accessible_node) <= self.num_tail_nodes and \
                path_length * self.num_nodes < \
                self.best_local_paths[-1]['shortest_length'] * (len(pre_path) + len(final_path) - 1):
            # 寻找下一个景点
            # return self.find_new_path(pre_path, final_path, path_length, bee, mode, accessible_node)
            return self.find_optimal_path(pre_path, final_path, path_length, bee, mode, accessible_node)
        else:
            # 寻找下一个景点
            return self.find_new_path(pre_path, final_path, path_length, bee, mode, accessible_node)

    # 确定性开发函数
    def find_optimal_path(self, pre_path, final_path, path_length, bee, mode, accessible_node):
        optimal_path = {'shortest_length': np.inf, 'shortest_path': []}
        final_path_tuple = tuple(final_path)
        tail_path_len = len(accessible_node) + 1
        if pre_path[-1] == 0:
            path = final_path
            ori_path_len = len(path)
            tmp_paths = set()

            # 执行广度优先搜索实现分枝定界
            tail_start = tuple(path[-1:])
            queue = deque([(tail_start, path_length)])
            tail_length = {(frozenset(tail_start[:-1]), tail_start[-1]): path_length}
            while queue:
                current_tail, current_length = queue.popleft()

                for next_node in accessible_node:
                    total_path_tuple = final_path_tuple[:-1] + current_tail + (next_node,)
                    if self.sca_mode and total_path_tuple in self.all_paths:
                        continue
                    if len(pre_path) == 1 and self.sca_mode:
                        tmp_paths.add(total_path_tuple)
                    current_tail_fset = frozenset(current_tail)
                    if next_node not in current_tail_fset:
                        new_tail = current_tail + (next_node,)
                        new_tail_set = (current_tail_fset, new_tail[-1])
                        new_length = current_length + self.distances[current_tail[-1]][next_node]
                        if new_tail_set in tail_length and tail_length[new_tail_set] <= new_length:
                            continue
                        tail_length[new_tail_set] = new_length
                        queue.append((new_tail, new_length))

                if len(current_tail) == tail_path_len:
                    new_tail = current_tail + tuple(pre_path)
                    new_length = current_length + self.distances[current_tail[-1]][pre_path[0]]
                    if new_length < optimal_path['shortest_length']:
                        optimal_path['shortest_length'] = new_length
                        optimal_path['shortest_path'] = new_tail

            self.all_paths.update(tmp_paths)
            final_path.extend(optimal_path['shortest_path'][1:])
            tail_path = final_path[ori_path_len - 1: ori_path_len + tail_path_len]

        else:
            path = pre_path
            pre_path_tuple = tuple(pre_path)
            record = len(accessible_node) + 1

            # 执行广度优先搜索实现分枝定界
            tail_start = tuple(path[-1:])
            queue = deque([(tail_start, path_length)])
            tail_length = {(frozenset(tail_start[:-1]), tail_start[-1]): path_length}
            while queue:
                current_tail, current_length = queue.popleft()

                for next_node in accessible_node:
                    current_tail_fset = frozenset(current_tail)
                    if next_node not in current_tail_fset:
                        new_tail = current_tail + (next_node,)
                        new_tail_set = (current_tail_fset, new_tail[-1])
                        new_length = current_length + self.distances[current_tail[-1]][next_node]
                        if new_tail_set in tail_length and tail_length[new_tail_set] <= new_length:
                            continue
                        tail_length[new_tail_set] = new_length
                        queue.append((new_tail, new_length))

                if len(current_tail) == tail_path_len:
                    new_length = current_length + self.distances[current_tail[-1]][pre_path[0]]
                    if new_length < optimal_path['shortest_length']:
                        for i in range(len(current_tail)):
                            if current_tail[i] == 0:
                                new_tail = current_tail[i:] + pre_path_tuple + current_tail[1:i + 1]
                                optimal_path['shortest_length'] = new_length
                                optimal_path['shortest_path'] = new_tail
                                record = i
                                break

            final_path.extend(optimal_path['shortest_path'][1:])
            tail_path = final_path[-record - 1:] + final_path[:len(accessible_node) + 1 - record]

        if len(final_path) == self.num_nodes + 1 and len(pre_path) != 1:
            # 执行尾部信息素更新
            self.tail_pheromone_update(tail_path, optimal_path['shortest_length'] - path_length)
            final_path_tuple = tuple(final_path[:-2])
            if final_path_tuple not in self.all_paths:
                if self.sca_mode:
                    self.all_paths.add(final_path_tuple)
                return optimal_path['shortest_length']
            path_length = optimal_path['shortest_length']
            path_length -= self.distances[final_path[-2]][final_path[-1]]
            pre_path = [final_path.pop()]
            path_length -= self.distances[final_path[-2]][final_path[-1]]
            accessible_node = set()
            accessible_node.add(final_path.pop())
            for i in range(1, len(final_path_tuple) - 1):
                path_length -= self.distances[final_path[-2]][final_path[-1]]
                accessible_node.add(final_path.pop())
                if final_path_tuple[:-i] not in self.all_paths:
                    if 0 < len(accessible_node) <= self.num_tail_nodes:
                        return self.find_optimal_path(pre_path, final_path, path_length, bee, mode, accessible_node)
                    else:
                        return self.find_new_path(pre_path, final_path, path_length, bee, mode, accessible_node)
            else:
                print(3)
                return path_length
        else:
            if self.sca_mode:
                self.all_paths.add(final_path_tuple)
            if len(final_path) == self.num_nodes + 1:
                return optimal_path['shortest_length']
            for i in range(1, len(final_path_tuple) - 1):
                path_length -= self.distances[final_path[-2]][final_path[-1]]
                accessible_node.add(final_path.pop())
                if final_path_tuple[:-i] not in self.all_paths:
                    if 0 < len(accessible_node) <= self.num_tail_nodes:
                        return self.find_optimal_path(pre_path, final_path, path_length, bee, mode, accessible_node)
                    else:
                        return self.find_new_path(pre_path, final_path, path_length, bee, mode, accessible_node)
            else:
                print(4)
                return path_length

    # 引力场更新函数
    def attractive_force_field_update(self, bee):
        self.attractive_force_field /= self.attractive_force_field
        for j in range(len(self.best_local_paths[-1]['shortest_path']) - 1):
            before = self.best_local_paths[-1]['shortest_path'][j]
            next = self.best_local_paths[-1]['shortest_path'][j + 1]
            self.attractive_force_field[before, next] = \
                1 + self.attraction_enhancement

        # for i in range(self.num_bees):
        #     self.bee_individual[i].alpha['developer'] = self.bee_individual[bee].alpha['developer']
        #     self.bee_individual[i].beta['developer'] = self.bee_individual[bee].beta['developer']
        #     self.bee_individual[i].q0['developer'] = self.bee_individual[bee].q0['developer']

    # 斥力场更新函数
    def repulsive_force_field_update(self, bee):
        edge = {}
        self.repulsive_force_field /= self.repulsive_force_field
        for i in range(len(self.best_local_paths)):
            for j in range(len(self.best_local_paths[i]['shortest_path']) - 1):
                before = self.best_local_paths[i]['shortest_path'][j]
                next = self.best_local_paths[i]['shortest_path'][j + 1]
                if (before, next) not in edge:
                    edge[(before, next)] = 0
                edge[(before, next)] += 1
                self.repulsive_force_field[before, next] = \
                    edge[(before, next)] / self.repulsion_enhancement

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
                1 / (self.distances[current_node][next_node] + 1e-8) / self.num_nodes)

    # 定义尾部信息素更新函数
    def tail_pheromone_update(self, tail_path, tail_length):
        # 遍历路径中的每段旅程
        for j in range(len(tail_path) - 1):
            self.pheromones[tail_path[j]][tail_path[j + 1]] *= (1 - self.tail_decay)
            # 增加新的信息素
            self.pheromones[tail_path[j]][tail_path[j + 1]] += \
                self.tail_decay * ((len(tail_path) - 1) / tail_length / self.num_nodes)

    def pso(self, mode, path_interests, iteration):
        # 一种改进粒子群算法
        # 找到最好的一个粒子，和次好的两个粒子，次好的两个粒子向最好的一个粒子移动；其余粒子向离这三个粒子中最近的一个粒子移动。
        # 参数边界处理：截断并反弹
        # 速度：考虑惯性、个体历史最佳位置，全局历史最佳位置
        # 全局最优个体参数不变
        def short(mode, best_individual, second_individual, third_individual, me):
            d = [0, 0, 0]
            d[0] = (best_individual.alpha[mode] - me.alpha[mode]) ** 2 + \
                   (best_individual.beta[mode] - me.beta[mode]) ** 2 + \
                   (best_individual.q0[mode] - me.q0[mode]) ** 2
            d[1] = (second_individual.alpha[mode] - me.alpha[mode]) ** 2 + \
                   (second_individual.beta[mode] - me.beta[mode]) ** 2 + \
                   (second_individual.q0[mode] - me.q0[mode]) ** 2
            d[2] = (third_individual.alpha[mode] - me.alpha[mode]) ** 2 + \
                   (third_individual.beta[mode] - me.beta[mode]) ** 2 + \
                   (third_individual.q0[mode] - me.q0[mode]) ** 2

            return d.index(min(d))

        def check(mode, me, local_mode):
            if local_mode == 0:
                # 边界限制与速度反向（反弹）：
                k = 0.05
                if me.alpha[mode] < 0.1:
                    me.alpha[mode] = 0.1
                    me.v[0][mode] = -me.v[0][mode] * k
                elif me.alpha[mode] > 2.0:
                    me.alpha[mode] = 2.0
                    me.v[0][mode] = -me.v[0][mode] * k

                if me.beta[mode] < 1.0:
                    me.beta[mode] = 1.0
                    me.v[1][mode] = -me.v[1][mode] * k
                elif me.beta[mode] > 5.0:
                    me.beta[mode] = 5.0
                    me.v[1][mode] = -me.v[1][mode] * k

                if me.q0[mode] < 0.1:
                    me.q0[mode] = 0.1
                    me.v[2][mode] = -me.v[2][mode] * k
                elif me.q0[mode] > 0.9:
                    me.q0[mode] = 0.9
                    me.v[2][mode] = -me.v[2][mode] * k

            if local_mode == 1:
                # 速度限制：
                k = 0.2
                while abs(me.v[0][mode]) > 0.85 * k:
                    me.v[0][mode] = me.v[0][mode] / abs(me.v[0][mode]) * 0.85 * k
                while abs(me.v[1][mode]) > 2.0 * k:
                    me.v[1][mode] = me.v[1][mode] / abs(me.v[1][mode]) * 2.0 * k
                while abs(me.v[2][mode]) > 0.40 * k:
                    me.v[2][mode] = me.v[2][mode] / abs(me.v[2][mode]) * 0.40 * k

        if iteration % 4 == 0:
            return
        for index, individual in enumerate(self.bee_individual):
            if individual.historic_best_fitness < path_interests[mode][index]:
                individual.historic_best_fitness = path_interests[mode][index]
                individual.historic_best_parameter[0] = individual.alpha[mode]
                individual.historic_best_parameter[1] = individual.beta[mode]
                individual.historic_best_parameter[2] = individual.q0[mode]
                individual.historic_best_parameter[3] = individual.start[mode]

        sorted_indices = sorted(range(self.num_bees // 2), key=lambda k: path_interests[mode][k], reverse=False)
        best_individual = self.bee_individual[sorted_indices[0]]
        second_individual = self.bee_individual[sorted_indices[1]]
        third_individual = self.bee_individual[sorted_indices[2]]
        best3_individual = [best_individual, second_individual, third_individual]

        best_individual.v = [{'explorer': 0, 'developer': 0}, {'explorer': 0, 'developer': 0},
                             {'explorer': 0, 'developer': 0}, {'explorer': 0, 'developer': 0}]  # 粒子移动速度

        # omega = 0.1 - (0.1* iteration) / (self.num_iterations)  # 非线性递减惯性权重
        # # omega = 0.5
        # c2 = 0.1+ iteration / self.num_iterations  # 群体经验项学习因子
        # c1 = 0.1- 0.1*iteration / self.num_iterations
        omega = 0.1  # 非线性递减惯性权重
        # omega = 0.5
        c2 = 0.1  # 群体经验项学习因子
        c1 = 0.1
        for index, i in enumerate(self.bee_individual):
            if index == sorted_indices[0]:
                continue
            k = short(mode, best_individual, second_individual, third_individual, i)
            if i == second_individual or i == third_individual:
                k = 0
            i.v[0][mode] = c2 * random.random() * (best3_individual[k].alpha[mode] - i.alpha[mode]) + \
                           c1 * random.random() * (i.historic_best_parameter[0] - i.alpha[mode]) + omega * i.v[0][mode]

            i.v[1][mode] = c2 * random.random() * (best3_individual[k].beta[mode] - i.beta[mode]) + \
                           c1 * random.random() * (i.historic_best_parameter[1] - i.beta[mode]) + omega * i.v[1][mode]
            i.v[2][mode] = c2 * random.random() * (best3_individual[k].q0[mode] - i.q0[mode]) + \
                           c1 * random.random() * (i.historic_best_parameter[2] - i.q0[mode]) + omega * i.v[2][mode]


            check(mode, i, 1)
            m = 0.75
            i.alpha[mode] += m * (1 - iteration / self.num_iterations) * i.v[0][mode]
            i.beta[mode] += m * (1 - iteration / self.num_iterations) * i.v[1][mode]
            i.q0[mode] += m * (1 - iteration / self.num_iterations) * i.v[2][mode]
            check(mode, self.bee_individual[index], 0)