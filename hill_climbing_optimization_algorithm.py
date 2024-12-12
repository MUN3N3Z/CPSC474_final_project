import random
import argparse
from my_policy import StatisticalThrower, RuleBasedPegger
from policy import CompositePolicy, GreedyThrower, GreedyPegger
from typing import List
from cribbage import Game, evaluate_policies
from json import dump
import time
import multiprocessing
import numpy as np
from functools import partial
ASSESSMENT_GAMES = 1000
CLIMBING_STEPS = 100

class HillClimbingAlgorithm:
    def __init__(self, time: int):
        self._parameters = [
            {'low': 0, 'high': 5},  # score_points
            {'low': 0, 'high': 2},  # lead_low_card
            {'low': 0, 'high': 2},  # lead_sum_to_15
            {'low': 0, 'high': 2},  # closest_to_31
            {'low': -2, 'high': 0},  # save_ace
            {'low': 0, 'high': 2},  # play_ace
            {'low': -2, 'high': 0},  # penalize_5
            {'low': -5, 'high': 0},  # illegal_play
        ]
        self._keys = [
            'score_points', 'lead_low_card', 'lead_sum_to_15', 
            'closest_to_31', 'save_ace', 'play_ace', 
            'penalize_5', 'illegal_play'
        ]
        self._training_time = time
        # Precompute number of cores for parallel evaluation
        self._num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    def _fitness(self, solution: np.ndarray) -> float:
        """
            - Run a cribbage agent using the weights from the solution in its rule-based pegging strategy and
            compare its performance against a baseline greedy pegger
            - Both agents will use the same statistical throwing strategy
        """
        
        card_weights = {key: value for key, value in zip(self._keys, solution)}
        game = Game()
        benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        hill_climbing_algorithm_policy = CompositePolicy(game, StatisticalThrower(game), RuleBasedPegger(game, card_weights))
        results = evaluate_policies(game, hill_climbing_algorithm_policy, benchmark, ASSESSMENT_GAMES)
        points = 0
        points = sum(match_value * number_of_games for match_value, number_of_games in results[3].items())
        return points / ASSESSMENT_GAMES
    
    def _parallel_fitness(self, solutions: List[np.ndarray]) -> List[float]:
        """
            - Evaluate the fitness of multiple solutions in parallel
        """
        fitness_func = partial(self._fitness)
        with multiprocessing.Pool(processes=self._num_cores) as pool:
            fitness_scores = pool.map(fitness_func, solutions)

        return fitness_scores
        
    def _generate_neighbor(self, solution: List[int]) -> List[int]:
        """
            - Generate a neighbor solution by randomly selecting a parameter and changing its value randomly by a small amount
        """
        neighbor = solution.copy()
        parameter_to_nudge_index = np.random.randint(0, len(self._parameters))
        parameter_space = self._parameters[parameter_to_nudge_index]
        parameter_step_size = (parameter_space['high'] - parameter_space['low']) / 10
        # Randomly decide whether to increase or decrease the parameter
        neighbor[parameter_to_nudge_index] += np.random.uniform(-parameter_step_size, parameter_step_size)
        # Ensure the new value is within the bounds
        neighbor[parameter_to_nudge_index] = np.clip(
            neighbor[parameter_to_nudge_index], 
            parameter_space['low'],
            parameter_space['high']
        )
        return neighbor
    
    def optimize(self) -> List[int]:
        """
            - Perform hill climbing optimization to find the best parameters for the rule-based pegger
        """
        best_parameter_weights, best_fitness = None, float('-inf')
        start_time = time.time()
        while time.time() - start_time < self._training_time:
            num_initial_solutions = self._num_cores * 2 # 2 solutions per core - increase the diversity of starting points
            initial_solutions = [np.array([np.random.uniform(parameter['low'], parameter['high']) for parameter in self._parameters]
                                                  ) for _ in range(num_initial_solutions)]
            initial_fitness_scores = self._parallel_fitness(initial_solutions)
            for solution, fitness in zip(initial_solutions, initial_fitness_scores):
                current_solution, current_fitness = solution, fitness
                for _ in range(CLIMBING_STEPS):
                    # Generate a neighbor solutions in parallel
                    num_neighbors = self._num_cores
                    neighbors = [self._generate_neighbor(current_solution) for _ in range(num_neighbors)]
                    neighbor_fitness_scores = self._parallel_fitness(neighbors)
                    # Find best neighbor
                    best_neighbor_index = np.argmax(neighbor_fitness_scores)
                    best_neighbor = neighbors[best_neighbor_index]
                    best_neighbor_fitness = neighbor_fitness_scores[best_neighbor_index]
                    # Update current solution if the neighbor is better
                    if best_neighbor_fitness > current_fitness:
                        current_solution, current_fitness = best_neighbor, best_neighbor_fitness
                # Update global best solution if current solution is bette
                if current_fitness > best_fitness:
                    best_parameter_weights, best_fitness = current_solution, current_fitness

        return best_parameter_weights, best_fitness
    
    def run(self):
        with open("hill_climbing_weights.json", "a") as f:
            best_weights, best_fitness = self.optimize()
            best_weights_dict = {key: value for key, value in zip(self._keys, best_weights)}
            dump(best_weights_dict, f, indent=4)
            print(f"Best fitness: {best_fitness}")
            

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--time", type=int, default=3600, help="The maximum time in seconds to train the hill climbing agent")
    args = arg_parser.parse_args()
    # Ensure multiprocessing is safe for spawning processes
    multiprocessing.set_start_method('spawn')
    hill_climbing_algorithm = HillClimbingAlgorithm(args.time)
    hill_climbing_algorithm.run()
