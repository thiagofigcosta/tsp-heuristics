import gc
import json
import os
import random as rd
import statistics
from enum import Enum
from time import perf_counter

import tsplib95

MAX_ITER_VNS = 1000000
MAX_ITER_VND = 10000  # ideally would be None = inf
MAX_ITER_GVNS = 100
MAX_ITER_VND_INSIDE_GVNS = MAX_ITER_VND // MAX_ITER_GVNS  # ideally would be None = inf
REPORT_PATH = 'res/report.json'
INSTANCE_FOLDER = 'res/'


class Solvers(Enum):
    BRUTE_FORCE = 0
    DYNAMIC_PROGRAMMING = 1
    CONSTRUCTION_HEURISTIC_NEAREST_NEIGHBOUR = 2
    CONSTRUCTION_HEURISTIC_REPETITIVE_NEAREST_NEIGHBOUR = 3
    HEURISTIC_RANDOM = 4
    HEURISTIC_REPETITIVE_RANDOM = 5
    CONSTRUCTION_HEURISTIC_CHEAPEST_LINK = 6
    GENERAL_VARIABLE_NEIGHBOURHOOD_SEARCH_HEURISTIC = 7
    VARIABLE_NEIGHBOURHOOD_DESCENT_HEURISTIC = 8
    VARIABLE_NEIGHBOURHOOD_SEARCH_HEURISTIC = 9
    GENETIC_METAHEURISTIC = 10
    CHRISTOFIDES_HEURISTIC = 11

    def __str__(self):
        return self.name

    @staticmethod
    def get_all_methods():
        return list(map(lambda c: c, Solvers))


def dummy_load_and_test_example(tsp_filepath):
    tsp_instance = tsplib95.load(tsp_filepath)
    nodes = list(tsp_instance.get_nodes())
    edges = list(tsp_instance.get_edges())
    edge = edges[1 if len(edges) > 1 else 0]
    distance = tsp_instance.get_weight(*edge)
    description = f'Distance from {edge[0]} to {edge[1]} is {distance}, there are {len(nodes)} cities.'
    print(description)


def _get_all_permutations_recursive(the_list, l, r, permutations):
    if l < r:
        _get_all_permutations_recursive(the_list, l + 1, r, permutations)
        for i in range(l + 1, r):
            the_list[l], the_list[i] = the_list[i], the_list[l]  # swap
            _get_all_permutations_recursive(the_list, l + 1, r, permutations)
            the_list[l], the_list[i] = the_list[i], the_list[l]  # swap
    else:
        permutations.append(the_list + [])  # copy list


def get_all_permutations(the_list):  # very memory consuming
    permutations = []
    _get_all_permutations_recursive(the_list, 0, len(the_list), permutations)
    return permutations


def _next_permutation_recursive(the_list, l, r):
    if l < r:
        yield from _next_permutation_recursive(the_list, l + 1, r)
        for i in range(l + 1, r):
            the_list[l], the_list[i] = the_list[i], the_list[l]  # swap
            yield from _next_permutation_recursive(the_list, l + 1, r)
            the_list[l], the_list[i] = the_list[i], the_list[l]  # swap
    else:
        yield the_list


def next_permutation(the_list):
    return _next_permutation_recursive(the_list, 0, len(the_list))


def random_permutation(the_list):
    permutation = the_list + []  # copy
    rd.shuffle(permutation)
    return permutation


def build_graph_from_edges(edges_list):
    nodes_set = set()
    graph_dict = {}
    total_cost = 0
    for edge in edges_list:
        if edge is None:
            continue
        cost, src, dst = edge
        total_cost += cost
        if src not in graph_dict:
            graph_dict[src] = set()
        if dst not in graph_dict:
            graph_dict[dst] = set()
        graph_dict[src].add(dst)
        graph_dict[dst].add(src)
        nodes_set.add(src)
        nodes_set.add(dst)
    return graph_dict, nodes_set, total_cost


def build_edge_and_adjacent_dicts(tsp_instance):
    edge_dict = {}
    adjacent_dict = {}
    for edge in tsp_instance.get_edges():
        source = edge[0]
        destination = edge[1]
        distance = tsp_instance.get_weight(*edge)
        edge_dict[edge] = distance
        if source not in adjacent_dict:
            adjacent_dict[source] = {}
        adjacent_dict[source][destination] = distance
    return edge_dict, adjacent_dict


def get_distance_from_edge_dict(edges, src, dst):
    return edges.get((src, dst,), float('inf'))


def get_distance_from_adjacent_dict(adjacent, src, dst):
    return adjacent.get(src, {dst: float('inf')})[dst]


def _solver_dynamic_programming_recursive(current_city_idx, mask, start_city, nodes, adjacent_dict, dp_memoization,
                                          path_tracker):
    current_city = nodes[current_city_idx]
    # mask stores the cities visited by the current path
    if mask == (1 << len(nodes)) - 1:  # all nodes were visited, starting from current_city
        return get_distance_from_adjacent_dict(adjacent_dict, current_city, start_city)  # return to beginning

    if dp_memoization[current_city_idx][mask] is not None:  # already computed
        # cost of shorter tour starting in current_city and going through the cities contained in the mask
        return dp_memoization[current_city_idx][mask]

    sub_problem_cost = float('inf')  # if not memoized, compute
    for idx, next_city in enumerate(nodes):
        if next_city != start_city and next_city != current_city:
            if (mask & (1 << idx)) == 0:  # if not visited next city yet
                cost_to_next_city = get_distance_from_adjacent_dict(adjacent_dict, current_city, next_city)
                mask_after_visiting_next_city = mask | (1 << idx)  # visit
                candidate_cost = cost_to_next_city + _solver_dynamic_programming_recursive(idx,
                                                                                           mask_after_visiting_next_city,
                                                                                           start_city, nodes,
                                                                                           adjacent_dict,
                                                                                           dp_memoization, path_tracker)
                if candidate_cost < sub_problem_cost:
                    sub_problem_cost = candidate_cost
                    path_tracker[current_city][mask] = next_city
    dp_memoization[current_city_idx][mask] = sub_problem_cost
    return sub_problem_cost


def _solver_dynamic_programming(tsp_instance):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]
    city_index_table = {city: i for i, city in enumerate(nodes)}

    starting_node = nodes[int(rd.random() * len(nodes))]
    mask_visited_starting_city = 1 << nodes.index(starting_node)

    path_tracker = {city: {} for city in nodes}
    # stores costs of tours from city (first dimension) to a set of nodes mask (2nd dimension)
    dp_memoization = [[None for _ in range(1 << len(nodes))] for _ in
                      range(len(nodes))]  # 2**10 == 1 << 10 == amount of permutations of 10 elements
    tour_cost = _solver_dynamic_programming_recursive(city_index_table[starting_node], mask_visited_starting_city,
                                                      starting_node, nodes, adjacent_dict, dp_memoization, path_tracker)

    tour = []
    cur_city = starting_node
    cur_mask = mask_visited_starting_city
    while True:
        tour.append(cur_city)
        cur_city = path_tracker[cur_city].get(cur_mask, None)
        if cur_city is None:
            break
        cur_mask = cur_mask | (1 << city_index_table[cur_city])
    tour.append(starting_node)

    return {
        'tour': tour,
        'cost': tour_cost
    }


def compute_cost(tour, adj_dict, is_origin_at_the_end=True):
    tour_cost = 0
    range_till = len(tour) - (1 if is_origin_at_the_end else 0)
    for i in range(range_till):
        current_city = tour[i]
        next_city = tour[i + 1 if i + 1 < len(tour) else 0]
        tour_cost += get_distance_from_adjacent_dict(adj_dict, current_city, next_city)
    return tour_cost


def compute_cost_ed(tour, edge_dict, is_origin_at_the_end=True):
    tour_cost = 0
    range_till = len(tour) - (1 if is_origin_at_the_end else 0)
    for i in range(range_till):
        current_city = tour[i]
        next_city = tour[i + 1 if i + 1 < len(tour) else 0]
        tour_cost += get_distance_from_edge_dict(edge_dict, current_city, next_city)
    return tour_cost


def _solver_brute_force(tsp_instance):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]

    solution = {
        'tour': [],
        'cost': float('inf')
    }
    for candidate_tour in next_permutation(nodes):
        candidate_tour_cost = compute_cost(candidate_tour, adjacent_dict, False)
        if candidate_tour_cost < solution['cost']:
            solution['cost'] = candidate_tour_cost
            solution['tour'] = candidate_tour + [candidate_tour[0]]  # list copy
    return solution


def _actual_solver_nearest_neighbour(nodes, adjacent_dict, starting_node=None):
    if starting_node is None:
        starting_node = nodes[int(rd.random() * len(nodes))]

    path = [starting_node]
    path_set = {starting_node}
    path_cost = 0

    while len(path) < len(nodes):
        candidate_neighbours = [(el[0], el[1],) for el in adjacent_dict[path[-1]].items()]
        candidate_neighbours.sort(key=lambda x: x[1])
        for next_city, cost in candidate_neighbours:
            if next_city != path[-1] and next_city not in path_set:
                path.append(next_city)
                path_set.add(next_city)
                path_cost += cost
                break
    path_cost += get_distance_from_adjacent_dict(adjacent_dict, path[-1], path[0])
    path.append(path[0])
    return {
        'tour': path,
        'cost': path_cost
    }


def _solver_nearest_neighbour(tsp_instance):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]
    return _actual_solver_nearest_neighbour(nodes, adjacent_dict)


def _solver_repetitive_nearest_neighbour(tsp_instance):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]
    solutions = []
    for node in nodes:
        solutions.append(_actual_solver_nearest_neighbour(nodes, adjacent_dict, starting_node=node))
    solutions.sort(key=lambda x: x['cost'])
    return solutions[0]


def _solver_random(tsp_instance):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]
    solution = random_permutation(nodes)  # this could be considered a constrution heuristic if I added one by one
    path = solution + [solution[0]]
    path_cost = compute_cost(path, adjacent_dict)
    return {
        'tour': path,
        'cost': path_cost
    }


def _solver_repetitive_random(tsp_instance, tries=None):
    if tries is None:
        tries = int(tsp_instance.dimension)
    solutions = []
    for _ in range(tries):
        solutions.append(_solver_random(tsp_instance))
    solutions.sort(key=lambda x: x['cost'])
    return solutions[0]


def circle_finder(graph_edges):
    def circle_finder_recursive(edges, visited_set, node, parent):
        visited_set.add(node)
        for neighbour in edges.get(node, []):
            if neighbour not in visited_set:
                if circle_finder_recursive(edges, visited_set, neighbour, node):
                    return True
            elif parent != neighbour:
                return True
        return False

    has_circle = False
    if len(graph_edges) == 1:
        return has_circle
    graph_dict, nodes_set, _ = build_graph_from_edges(graph_edges)
    visited = set()
    for n in nodes_set:
        if n not in visited:
            if circle_finder_recursive(graph_dict, visited, n, None):
                has_circle = True
                break

    return has_circle


def get_canonical_edge(cost, src, dst):
    if src < dst:
        edge = f'{src}|{dst}|{cost}'
    else:
        edge = f'{dst}|{src}|{cost}'
    return edge


def build_minimum_spamming_tree(nodes, edge_list, degree_restriction=float('inf')):
    edge_list.sort(key=lambda x: x[0])  # sort by cheapest
    degrees = {node: 0 for node in nodes}
    canonical_edges_added = set()  # avoid inserting the same edge reversed

    minimum_spamming_tree = []
    edge_index = 0  # to iterate from near to far edges
    for _ in range(len(nodes) - 1):
        minimum_spamming_tree.append(None)
        added = False
        while not added and edge_index < len(edge_list):
            edge = edge_list[edge_index]
            cost, src, dst = edge
            canonical_edge = get_canonical_edge(cost, src, dst)
            if degrees[src] < degree_restriction and degrees[dst] < degree_restriction and \
                    canonical_edge not in canonical_edges_added:  # no city can be connected to more than 2 others
                minimum_spamming_tree[-1] = edge
                has_circle = circle_finder(
                    minimum_spamming_tree)  # if the path don't make a cycle we can add
                if not has_circle:
                    added = True
                    degrees[src] += 1
                    degrees[dst] += 1
                    canonical_edges_added.add(canonical_edge)
            edge_index += 1

    return minimum_spamming_tree, degrees


def dfs(graph, start):
    path = []
    visited = set()
    to_visit = [start]  # stack = depth first search | queue = breadth first search
    while len(to_visit) > 0:
        visiting = to_visit[-1]  # get last = stack
        to_visit = to_visit[:-1]  # remove last = stack
        if visiting not in visited:
            visited.add(visiting)
            path.append(visiting)
            for neighbour in graph.get(visiting, []):
                to_visit.append(neighbour)  # add to end = stack
    return path


def _solver_cheapest_link(tsp_instance):  # almost a Christofides
    nodes = list(tsp_instance.get_nodes())
    edge_dict = build_edge_and_adjacent_dicts(tsp_instance)[0]

    edge_list = [[cost, src, dst] for (src, dst), cost in edge_dict.items() if src != dst]  # list only non self connect
    path_edges, degree = build_minimum_spamming_tree(nodes, edge_list, degree_restriction=2)

    # add the final edge, or I could keep iterating the nodes before and allow circle if last edge
    final_src, final_dst = [x[0] for x in filter(lambda x: x[1] < 2, degree.items())]
    path_edges.append([get_distance_from_edge_dict(edge_dict, final_src, final_dst), final_src, final_dst])

    # find the tour
    path_dict, _, path_cost = build_graph_from_edges(path_edges)
    path = dfs(path_dict, final_src)
    path.append(path[0])

    return {
        'tour': path,
        'cost': path_cost
    }


def get_n_random_indexes(tour, n, is_origin_at_the_end=True):
    get_random = lambda: int(rd.random() * (len(tour) - (2 if is_origin_at_the_end else 1))) + (
        0 if is_origin_at_the_end else 1)
    indexes = set()
    if n > len(tour) - 2:
        raise Exception()
    while n != len(indexes):
        indexes.add(get_random())
    indexes = list(indexes)
    indexes.sort()
    return indexes


def perform_swap(tour, i, j, copy=True):
    changed = tour + [] if copy else tour
    changed[i], changed[j] = changed[j], changed[i]
    return changed


def perform_2opt(tour, i, j, copy=True):
    changed = tour + [] if copy else tour
    changed[i:j + 1] = reversed(changed[i:j + 1])
    return changed


def random_swap(tour, is_origin_at_the_end=True, copy=True):
    i, j = get_n_random_indexes(tour, 2, is_origin_at_the_end=is_origin_at_the_end)
    return perform_swap(tour, i, j, copy=copy)


def random_2opt(tour, is_origin_at_the_end=True, copy=True):
    i, j = get_n_random_indexes(tour, 2, is_origin_at_the_end=is_origin_at_the_end)
    return perform_2opt(tour, i, j, copy=copy)


def best_swap(tour, adj_dict, is_origin_at_the_end=True, copy=True):
    amount_nodes = len(tour) - (1 if is_origin_at_the_end else 0)
    candidates = []
    for i in range((1 if is_origin_at_the_end else 0), amount_nodes, 1):
        for j in range(i, amount_nodes, 1):
            candidates.append((compute_cost(perform_swap(tour, i, j, True), adj_dict, is_origin_at_the_end), (i, j,),))
    candidates.sort(key=lambda x: x[0])
    return perform_swap(tour, candidates[0][1][0], candidates[0][1][1], copy=copy)


def best_2opt(tour, adj_dict, is_origin_at_the_end=True, copy=True, allow_change_on_first=False):
    amount_nodes = len(tour) - (1 if is_origin_at_the_end else 0)
    candidates = []
    for i in range((1 if is_origin_at_the_end else 0), amount_nodes, 1):
        for j in range(i, amount_nodes, 1):
            candidates.append((compute_cost(perform_2opt(tour, i, j, True), adj_dict, is_origin_at_the_end), (i, j,),))
    candidates.sort(key=lambda x: x[0])
    return perform_2opt(tour, candidates[0][1][0], candidates[0][1][1], copy=copy)


class NeighbourhoodMethod(Enum):
    RANDOM_SWAP = 0
    RANDOM_2OPT = 1
    BEST_SWAP = 2
    BEST_2OPT = 3

    def __str__(self):
        return self.name

    @staticmethod
    def get_all_methods():
        return list(map(lambda c: c, NeighbourhoodMethod))


def get_neighbour(neighbourhood_method, initial, adjacent_dict):
    neighbour = None
    if neighbourhood_method == NeighbourhoodMethod.RANDOM_SWAP:
        neighbour = random_swap(initial, False)
    elif neighbourhood_method == NeighbourhoodMethod.RANDOM_2OPT:
        neighbour = random_2opt(initial, False)
    elif neighbourhood_method == NeighbourhoodMethod.BEST_SWAP:
        neighbour = best_swap(initial, adjacent_dict, False)
    elif neighbourhood_method == NeighbourhoodMethod.BEST_2OPT:
        neighbour = best_2opt(initial, adjacent_dict, False)
    return neighbour


def range_infinite():
    very_patient_counter = 0
    while True:
        yield very_patient_counter
        very_patient_counter += 1


def _vnd_core(solution, adjacent_dict, neighbourhood_methods=None, max_iter=None, max_patience=None):
    if neighbourhood_methods is None:
        neighbourhood_methods = [NeighbourhoodMethod.BEST_SWAP, NeighbourhoodMethod.BEST_2OPT]
    solution_cost = compute_cost(solution, adjacent_dict, False)
    patience = len(neighbourhood_methods) if max_patience is None else max_patience
    for i in range(max_iter) if max_iter is not None else range_infinite():
        neighbourhood_method = neighbourhood_methods[i % len(neighbourhood_methods)]
        neighbour = get_neighbour(neighbourhood_method, solution, adjacent_dict)
        neighbour_cost = compute_cost(neighbour, adjacent_dict, False)
        if neighbour_cost < solution_cost:
            solution = neighbour
            solution_cost = neighbour_cost
            patience = len(neighbourhood_methods)
        else:
            patience -= 1
        if patience == 0:  # like the teacher reading this messy code
            break
    return solution, solution_cost


def _solver_vnd(tsp_instance, max_iter=None):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]

    initial = random_permutation(nodes)
    solution, cost = _vnd_core(initial, adjacent_dict, max_iter=max_iter)
    path = solution + [solution[0]]
    return {
        'tour': path,
        'cost': cost
    }


def _solver_gvns(tsp_instance, max_iter=50, vnd_max_iter=None, max_patience=15):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]

    neighbourhood_methods = [NeighbourhoodMethod.RANDOM_SWAP, NeighbourhoodMethod.RANDOM_2OPT]
    solution = random_permutation(nodes)
    solution_cost = compute_cost(solution, adjacent_dict, False)
    patience = max_patience
    for i in range(max_iter):  # Termination condition
        while patience > 0:
            neighbourhood_method = neighbourhood_methods[(i + patience) % len(neighbourhood_methods)]
            # I don't need a counter, since I don't have a list of neighbours
            neighbour = get_neighbour(neighbourhood_method, solution, adjacent_dict)  # shake
            neighbour, neighbour_cost = _vnd_core(neighbour, adjacent_dict, NeighbourhoodMethod.get_all_methods(),
                                                  max_iter=vnd_max_iter, max_patience=vnd_max_iter)
            if neighbour_cost < solution_cost:
                solution = neighbour
                solution_cost = neighbour_cost
                patience = max_patience
            else:
                patience -= 1
            if patience == 0:  # like the teacher reading this messy code
                break

    path = solution + [solution[0]]
    return {
        'tour': path,
        'cost': solution_cost
    }


def _solver_vns(tsp_instance, max_iter=50, max_patience=50):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]

    neighbourhood_methods = [NeighbourhoodMethod.RANDOM_SWAP, NeighbourhoodMethod.RANDOM_2OPT]
    solution = random_permutation(nodes)
    solution_cost = compute_cost(solution, adjacent_dict, False)
    patience = max_patience
    for i in range(max_iter):  # Termination condition
        while patience > 0:
            neighbourhood_method = neighbourhood_methods[(i + patience) % len(neighbourhood_methods)]
            # I don't need a counter, since I don't have a list of neighbours
            neighbour = get_neighbour(neighbourhood_method, solution, adjacent_dict)  # shake
            neighbour_cost = compute_cost(neighbour, adjacent_dict, False)
            if neighbour_cost < solution_cost:
                solution = neighbour
                solution_cost = neighbour_cost
                patience = max_patience
            else:
                patience -= 1
            if patience == 0:  # like the teacher reading this messy code
                break
    path = solution + [solution[0]]
    return {
        'tour': path,
        'cost': solution_cost
    }


def _solver_genetic_algorithm(tsp_instance, gens=666, pop_size=666, love_rate=.6, radiation_rate=.3):
    nodes = list(tsp_instance.get_nodes())
    adjacent_dict = build_edge_and_adjacent_dicts(tsp_instance)[1]
    pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1

    nodes_set = set(nodes)
    population = []
    for _ in range(pop_size):  # initializing
        population.append({'dna': random_permutation(nodes), 'fitness': float('inf')})

    solution = None
    solution_cost = float('inf')
    for g in range(gens + 1):
        for ind in population:
            ind['fitness'] = compute_cost(ind['dna'], adjacent_dict, False)
            if ind['fitness'] < solution_cost:
                solution_cost = ind['fitness']
                solution = ind['dna']
        if g < gens:
            # select
            population.sort(key=lambda x: x['fitness'])
            min_fit = population[0]['fitness']
            offset = 0
            if min_fit < 0:
                offset = abs(min_fit)
            fitness_sum = 0
            for ind in population:
                fitness_sum += ind['fitness']
            next_gen = []
            for _ in range(len(population) // 2):
                parents = []
                for c in range(2):
                    roulette = rd.random() * fitness_sum
                    sphere_position = 0
                    for ind in population:
                        sphere_position += ind['fitness'] + offset
                        if sphere_position > roulette:
                            parents.append(ind)
                            break
                # crossover
                # https://link.springer.com/article/10.1007/BF02125403 has better ways to do this, but I'm creating mine
                if rd.random() < love_rate:
                    a, b = get_n_random_indexes(parents[0]['dna'], 2, True)  # could be any odd number
                    children = [{'dna': parents[0]['dna'][:a] + parents[1]['dna'][a:b] + parents[0]['dna'][b:],
                                 'fitness': float('inf')},
                                {'dna': parents[1]['dna'][:a] + parents[0]['dna'][a:b] + parents[1]['dna'][b:],
                                 'fitness': float('inf')}]
                    for child in children:
                        missing_nodes = random_permutation(list(nodes_set.difference(child['dna'])))
                        if len(missing_nodes) > 0:
                            duplicated_indexes = [_i for _i, _el in enumerate(child['dna']) if _el in child['dna'][:_i]]
                            for i, missing in zip(duplicated_indexes, missing_nodes):
                                child['dna'][i] = missing
                    next_gen += children
                else:
                    next_gen += parents
            # mutation
            for ind in population:
                if rd.random() < radiation_rate:
                    ind['dna'] = random_2opt(ind['dna'], False, False)  # inplace but ill do the assign anyway

    path = solution + [solution[0]]
    return {
        'tour': path,
        'cost': solution_cost
    }


def get_eulerian_path_recursive(current, graph, euler_path):
    def can_move(src, dst, _graph):
        if dst not in _graph[src] or src not in _graph[dst]:
            return False
        if len(_graph[src]) == 1:
            return True
        reachable_nodes_w_edge = len(dfs(_graph, src)) - 1
        _graph[src].remove(dst)
        _graph[dst].remove(src)
        reachable_nodes_wo_edge = len(dfs(_graph, src)) - 1
        _graph[src].add(dst)
        _graph[dst].add(src)
        return reachable_nodes_w_edge <= reachable_nodes_wo_edge

    for neighbour in list(graph[current]):
        if neighbour not in graph[current]:
            continue
        if can_move(current, neighbour, graph):
            graph[neighbour].remove(current)
            graph[current].remove(neighbour)
            euler_path.append(neighbour)
            get_eulerian_path_recursive(neighbour, graph, euler_path)


def get_eulerian_path_fleury_algo_BROKE(graph):  # TODO fix me
    graph_deep_deep_copy = {k: set(list(v)) for k, v in
                            graph.items()}  # copy.copy / copy.deepcopy not working as expected, copy makes a shallow copy and deep copy causes the sets to be empty
    starting = next(iter(graph_deep_deep_copy.keys()))
    euler_path = [starting]
    for node, neighbours in graph_deep_deep_copy.items():
        if len(neighbours) % 2 == 1:
            starting = node
            break
    get_eulerian_path_recursive(starting, graph_deep_deep_copy, euler_path)
    return euler_path


def get_eulerian_path(graph):
    graph_deep_deep_copy = {k: set(list(v)) for k, v in
                            graph.items()}  # copy.copy / copy.deepcopy not working as expected, copy makes a shallow copy and deep copy causes the sets to be empty

    starting = next(iter(graph_deep_deep_copy.keys()))
    euler_path = [starting]

    while any([len(v) > 0 for v in graph_deep_deep_copy.values()]):
        src_node = None
        current_path_idx = None
        for idx, node in enumerate(euler_path):
            if len(graph_deep_deep_copy[node]) > 0:
                src_node = node
                current_path_idx = idx
                break
        while src_node is not None and len(graph_deep_deep_copy[src_node]) > 0:
            dst_node = next(iter(graph_deep_deep_copy[src_node]))
            euler_path.insert(current_path_idx, dst_node)
            current_path_idx += 1
            graph_deep_deep_copy[src_node].remove(dst_node)
            graph_deep_deep_copy[dst_node].remove(src_node)
    return euler_path


def _solver_christofides(tsp_instance):
    nodes = list(tsp_instance.get_nodes())
    edge_dict = build_edge_and_adjacent_dicts(tsp_instance)[0]

    edge_list = [[cost, src, dst] for (src, dst), cost in edge_dict.items() if src != dst]  # list only non self connect
    mst, degrees = build_minimum_spamming_tree(nodes, edge_list)

    odd_vertices = [x[0] for x in filter(lambda x: x[1] % 2 == 1, degrees.items())]

    minimum_weight_matching = []
    rd.shuffle(odd_vertices)
    while len(odd_vertices) > 0:
        from_vertice = odd_vertices[-1]
        odd_vertices = odd_vertices[:-1]
        best_edge = [[float('inf'), from_vertice, from_vertice], -1]
        for idx, to_vertice in enumerate(odd_vertices):
            distance = get_distance_from_edge_dict(edge_dict, from_vertice, to_vertice)
            if distance < best_edge[0][0]:
                best_edge = [[distance, from_vertice, to_vertice], idx]
        minimum_weight_matching.append(best_edge[0])
        odd_vertices.pop(best_edge[1])

    eulerian_graph_edges = mst + minimum_weight_matching
    eulerian_graph = build_graph_from_edges(eulerian_graph_edges)[0]
    eulerian_path = get_eulerian_path(eulerian_graph)

    seen_nodes = set()
    hamiltonian_tour = [el for el in eulerian_path if el not in seen_nodes and not seen_nodes.add(el)] + [
        eulerian_path[0]]
    cost = compute_cost_ed(hamiltonian_tour, edge_dict)
    return {
        'tour': hamiltonian_tour,
        'cost': cost
    }


def solve_tsp_instance(tsp_instance, solver):
    solution = {}
    start_time = perf_counter()
    if solver == Solvers.BRUTE_FORCE:
        solution = _solver_brute_force(tsp_instance)
    elif solver == Solvers.DYNAMIC_PROGRAMMING:
        solution = _solver_dynamic_programming(tsp_instance)
    elif solver == Solvers.CONSTRUCTION_HEURISTIC_NEAREST_NEIGHBOUR:
        solution = _solver_nearest_neighbour(tsp_instance)
    elif solver == Solvers.CONSTRUCTION_HEURISTIC_REPETITIVE_NEAREST_NEIGHBOUR:
        solution = _solver_repetitive_nearest_neighbour(tsp_instance)
    elif solver == Solvers.HEURISTIC_RANDOM:
        solution = _solver_random(tsp_instance)
    elif solver == Solvers.HEURISTIC_REPETITIVE_RANDOM:
        solution = _solver_repetitive_random(tsp_instance)
    elif solver == Solvers.CONSTRUCTION_HEURISTIC_CHEAPEST_LINK:
        solution = _solver_cheapest_link(tsp_instance)
    elif solver == Solvers.VARIABLE_NEIGHBOURHOOD_DESCENT_HEURISTIC:
        solution = _solver_vnd(tsp_instance, max_iter=MAX_ITER_VND)
    elif solver == Solvers.GENERAL_VARIABLE_NEIGHBOURHOOD_SEARCH_HEURISTIC:
        solution = _solver_gvns(tsp_instance, max_iter=MAX_ITER_GVNS, vnd_max_iter=MAX_ITER_VND_INSIDE_GVNS)
    elif solver == Solvers.VARIABLE_NEIGHBOURHOOD_SEARCH_HEURISTIC:
        solution = _solver_vns(tsp_instance, max_iter=MAX_ITER_VNS)
    elif solver == Solvers.GENETIC_METAHEURISTIC:
        solution = _solver_genetic_algorithm(tsp_instance)
    elif solver == Solvers.CHRISTOFIDES_HEURISTIC:
        solution = _solver_christofides(tsp_instance)
    delta_s = perf_counter() - start_time
    solution['runtime_s'] = delta_s
    missing_nodes = (len(solution['tour']) - 1) < len(list(tsp_instance.get_nodes()))
    extra_nodes = (len(solution['tour']) - 1) > len(list(tsp_instance.get_nodes()))
    duplicates = len(solution['tour'][:-1]) != len(set(solution['tour']))
    none = any(x is None for x in solution['tour'])
    if missing_nodes or duplicates or extra_nodes or none:
        solution['errors'] = ''
        if missing_nodes:
            solution['errors'] += 'missing nodes, '
        if extra_nodes:
            solution['errors'] += 'extra nodes, '
        if duplicates:
            solution['errors'] += 'duplicates, '
        if none:
            solution['errors'] += 'none, '
        print(f"The instance {tsp_instance.name} had the following ERRORS: {solution['errors']}")
    return solution


def test(solver=Solvers.BRUTE_FORCE):
    tsp_filepath = 'res/EUC_2D/dummy5.tsp'
    dummy_load_and_test_example(tsp_filepath)
    solution = solve_tsp_instance(tsplib95.load(tsp_filepath), solver)
    print(solution)
    exit()


def get_stats(array):
    r_mean = None
    r_median = None
    r_stdev = None
    try:
        r_mean = statistics.mean(array)
    except:
        pass
    try:
        r_median = statistics.median(array)
    except:
        pass
    try:
        r_stdev = statistics.stdev(array)
    except:
        pass
    return r_mean, r_median, r_stdev


def save_report(rep):
    for met, met_data in rep.items():
        if '$' in met:
            continue
        for inst, sol in met_data['solutions'].items():
            if '$' in inst:
                continue
            rep[met]['solutions'][inst]['sol_quality'] = round(
                rep['$Known_best_solutions'][inst][0] / sol['cost'] * 100,
                2)
    for inst, data in rep['$Known_best_solutions'].items():
        rep['$Known_best_solutions'][inst] = json.dumps(data).replace('\"', "'")
    try:
        with open(REPORT_PATH, 'w') as f:
            json.dump(rep, f, ensure_ascii=False, indent=3)
    except Exception as e:
        print(e)
    for inst, data in rep['$Known_best_solutions'].items():
        rep['$Known_best_solutions'][inst] = json.loads(data.replace("'", '\"'))


# test(solver=Solvers.BRUTE_FORCE)


all_tsp_instance_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(INSTANCE_FOLDER) for f in filenames if
                          os.path.splitext(f)[1] == '.tsp']

all_tsp_instance_files = [(tsplib95.load(x).dimension, x) for x in all_tsp_instance_files]
all_tsp_instance_files.sort(key=lambda x: x[0])  # slowest later
all_tsp_instance_files = [x[1] for x in all_tsp_instance_files]

all_solving_methods = Solvers.get_all_methods()
all_solving_methods.reverse()  # slowest later

ultra_slow_methods = {Solvers.BRUTE_FORCE, Solvers.DYNAMIC_PROGRAMMING}
slow_methods = set([Solvers.VARIABLE_NEIGHBOURHOOD_DESCENT_HEURISTIC,
                    Solvers.GENERAL_VARIABLE_NEIGHBOURHOOD_SEARCH_HEURISTIC, Solvers.GENETIC_METAHEURISTIC] + list(
    ultra_slow_methods))

TEST_ALL_METHODS = False

min_values = {k: [float('inf'), []] for k in all_tsp_instance_files}
report = {'$Known_best_solutions': min_values}
total_runtime = 0
for method in all_solving_methods:
    runtimes = []
    method_name = str(method)
    report[method_name] = {'solutions': {}}
    print(f'Running {method}...')
    for i, instance_file in enumerate(all_tsp_instance_files):
        if TEST_ALL_METHODS and 'dummy' not in instance_file:
            continue
        if method in slow_methods:  # slow methods
            print(f"File {i + 1} out of {len(all_tsp_instance_files)}...")
        if method in ultra_slow_methods and 'dummy' not in instance_file:
            print('Skipping files, since this is a ultra slow method...')
            break
        try:
            solution = solve_tsp_instance(tsplib95.load(instance_file), method)
            solution['tour'] = str(solution['tour'])
            report[method_name]['solutions'][instance_file] = solution
            if 'errors' not in solution:
                if solution['cost'] < min_values[instance_file][0]:
                    min_values[instance_file][0] = solution['cost']
                    min_values[instance_file][1] = [method_name]
                elif solution['cost'] == min_values[instance_file][0]:
                    min_values[instance_file][1].append(method_name)
            if 'dummy' not in instance_file:
                runtimes.append(solution['runtime_s'])
        except Exception as e:
            print(e)
            raise e
        if method in slow_methods:
            save_report(report)
        gc.collect()

    total_runtime += sum(runtimes)
    report['$Total_runtime'] = total_runtime
    r_mean, r_median, r_stdev = get_stats(runtimes)
    report[method_name]['$Runtimes'] = {
        'runtimes': runtimes,
        'mean': r_mean,
        'median': r_median,
        'stdev': r_stdev,
    }
    save_report(report)
    gc.collect()

print('Happy end')
