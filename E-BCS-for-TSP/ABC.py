import numpy as np
import random
from itertools import permutations

# 计算距离
def distance_matrix(cities):
    num_cities = len(cities)
    dm = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dm[i][j] = np.linalg.norm(cities[i] - cities[j])
    return dm

# 路径构建阶段
def pathConstruction(distance_matrix, alpha, beta, rho, Q, a, b):
    num_cities = len(distance_matrix)
    num_ants = 5  # 蚂蚁数量，可以根据需要调整
    pheromone_matrix = np.ones((num_cities, num_cities)) * (1 / num_cities)  # 信息素矩阵初始化
    best_path = None
    best_path_length = float('inf')

    for ant in range(num_ants):
        current_city = 0
        visited = [current_city]
        path = [current_city]
        path_length = 0

        while len(visited) < num_cities:
            next_city_probabilities = np.zeros(num_cities)
            for j in range(num_cities):
                if j not in visited:
                    # 计算选择概率
                    prob = (pheromone_matrix[current_city][j] ** alpha) * ((1 / distance_matrix[current_city][j]) ** beta)
                    next_city_probabilities[j] = prob
            # 选择下一个城市
            next_city = np.random.choice(num_cities, p=next_city_probabilities / np.sum(next_city_probabilities))
            visited.append(next_city)
            path.append(next_city)
            path_length += distance_matrix[current_city][next_city]
            current_city = next_city

        # 更新最佳路径
        if path_length < best_path_length:
            best_path = path
            best_path_length = path_length

        # 更新信息素
        for i in range(len(path) - 1):
            pheromone_matrix[path[i]][path[i + 1]] += Q / distance_matrix[path[i]][path[i + 1]]

    return best_path, best_path_length

# 路径改进阶段
def pathImprovement(best_path, distance_matrix, a, b, rho):
    num_cities = len(best_path)
    num_improvements = 5  # 改进次数，可以根据需要调整
    best_path_length = np.inf

    for _ in range(num_improvements):
        new_path = best_path[:]
        new_path_length = 0
        for i in range(num_cities - 1):
            # 随机选择一个点进行插入
            random_point = random.randint(0, num_cities - 1)
            random_element = best_path[random_point]
            # 移除并插入到当前位置
            new_path = new_path[:i + 1] + [random_element] + new_path[i + 1:]
            new_path_length = distance_matrix[best_path[i]][random_element] + distance_matrix[random_element][best_path[i + 1]]

        if new_path_length < best_path_length:
            best_path = new_path
            best_path_length = new_path_length

    return best_path, best_path_length


import random


def path_improvement(best_path, distance_matrix, num_improvements):
    for _ in range(num_improvements):
        operation = random.choice(['RI', 'RIS', 'RRIS'])
        if operation == 'RI':
            new_path = random_insertion_of_point(best_path, distance_matrix)
        elif operation == 'RIS':
            new_path = random_insertion_of_subsequences(best_path, distance_matrix)
        elif operation == 'RRIS':
            new_path = reverse_random_insertion_of_subsequences(best_path, distance_matrix)

        # 计算新路径的长度
        new_path_length = calculate_path_length(new_path, distance_matrix)

        # 如果新路径更优，则接受新路径
        if new_path_length < calculate_path_length(best_path, distance_matrix):
            best_path = new_path

    return best_path


def random_insertion_of_point(path, distance_matrix):
    # 选择一个点并移除它
    point_to_insert = random.choice(range(len(path)))
    path = [p for p in path if p != point_to_insert]

    # 随机选择插入位置
    insert_position = random.choice(range(len(path) + 1))

    # 将点插入到路径中
    new_path = path[:insert_position] + [point_to_insert] + path[insert_position:]

    return new_path


def random_insertion_of_subsequences(path, distance_matrix):
    # 选择一个子序列并移除它
    start = random.choice(range(len(path)))
    end = start + 1
    while end < len(path) and path[end] != path[start]:
        end += 1
    subsequence = path[start:end]

    # 移除子序列
    path = path[:start] + path[end:]

    # 随机选择插入位置
    insert_position = random.choice(range(len(path) + 1))

    # 将子序列插入到路径中
    new_path = path[:insert_position] + subsequence + path[insert_position:]

    return new_path


def reverse_random_insertion_of_subsequences(path, distance_matrix):
    # 选择一个子序列并移除它
    start = random.choice(range(len(path)))
    end = start + 1
    while end < len(path) and path[end] != path[start]:
        end += 1
    subsequence = list(reversed(path[start:end]))

    # 移除子序列
    path = path[:start] + path[end:]

    # 随机选择插入位置
    insert_position = random.choice(range(len(path) + 1))

    # 将子序列插入到路径中
    new_path = path[:insert_position] + subsequence + path[insert_position:]

    return new_path


def calculate_path_length(path, distance_matrix):
    length = 0
    for i in range(len(path) - 1):
        length += distance_matrix[path[i]][path[i + 1]]
    length += distance_matrix[path[-1]][path[0]]  # 添加回到起点的距离
    return length


# 示例使用
if __name__ == "__main__":
    # 假设的距离矩阵
    distance_matrix = [
        [0, 1, 2, 3],
        [1, 0, 5, 6],
        [2, 5, 0, 8],
        [3, 6, 8, 0]
    ]
    # 假设的最佳路径
    best_path = [0, 1, 2, 3]
    # 改进路径
    improved_path = path_improvement(best_path, distance_matrix, 100)
    print("Improved Path:", improved_path)


# 主函数
def main():
    cities = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # 示例城市坐标
    distancematrix = distance_matrix(cities)
    alpha = 1.0  # 信息素重要度
    beta = 5.0  # 启发式信息重要度
    rho = 0.65  # 信息素挥发系数
    Q = 100  # 信息素强度
    a = 1.0  # 足迹机制参数
    b = 5.0  # 足迹机制参数

    best_path, best_path_length = pathConstruction(distancematrix, alpha, beta, rho, Q, a, b)
    print("Best Path from Construction:", best_path, "Length:", best_path_length)

    best_path, best_path_length = pathImprovement(best_path, distancematrix, a, b, rho)
    print("Best Path after Improvement:", best_path, "Length:", best_path_length)

if __name__ == "__main__":
    main()