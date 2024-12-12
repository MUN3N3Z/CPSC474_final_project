import random
from my_policy import StatisticalThrower, RuleBasedPegger
from policy import CompositePolicy, GreedyThrower, GreedyPegger
from typing import List
from cribbage import Game, evaluate_policies
from json import dump

MAX_ITERTATIONS = 1000
ASSESSMENT_GAMES = 1000

class HillClimbingAlgorithm:
    def __init__(self):
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
    
    def _fitness(self, solution: List[int]) -> float:
        """
            - Run a cribbage agent using the weights from the solution in its rule-based pegging strategy and
            compare its performance against a baseline greedy pegger
            - Both agents will use the same statistical throwing strategy
        """
        
        card_weights = {key: value for key, value in zip(self._keys, solution)}
        game = Game()
        benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        genetic_algorithm_policy = CompositePolicy(game, StatisticalThrower(game), RuleBasedPegger(game, card_weights))
        results = evaluate_policies(game, genetic_algorithm_policy, benchmark, ASSESSMENT_GAMES)
        points = 0
        for match_value, number_of_games in results[3].items():
            points += match_value * match_value * number_of_games
        return points / ASSESSMENT_GAMES
    
    def _generate_neighbor(self, solution: List[int]) -> List[int]:
        """
            - Generate a neighbor solution by randomly selecting a parameter and changing its value randomly by a small amount
        """
        neighbor = solution.copy()
        parameter_to_nudge_index = random.randint(0, len(solution) - 1)
        parameter_space = self._parameters[parameter_to_nudge_index]
        parameter_step_size = (parameter_space['high'] - parameter_space['low']) / 10
        # Randomly decide whether to increase or decrease the parameter
        if random.random() > 0.5:
            neighbor[parameter_to_nudge_index] += parameter_step_size
        else:
            neighbor[parameter_to_nudge_index] -= parameter_step_size
        # Ensure the new value is within the bounds
        neighbor[parameter_to_nudge_index] = max(parameter_space['low'], min(parameter_space['high'], neighbor[parameter_to_nudge_index]))
        return neighbor
    
    def optimize(self) -> List[int]:
        """
            - Perform hill climbing optimization to find the best parameters for the rule-based pegger
        """
        current_parameter_weights = [random.uniform(parameter['low'], parameter['high']) for parameter in self._parameters]
        current_fitness = self._fitness(current_parameter_weights)
        for _ in range(MAX_ITERTATIONS):
            neighbor_parameter_weights = self._generate_neighbor(current_parameter_weights)
            neighbor_fitness = self._fitness(neighbor_parameter_weights)
            current_fitness, current_parameter_weights = (neighbor_fitness, neighbor_parameter_weights) if neighbor_fitness > current_fitness else (current_fitness, current_parameter_weights)
        return current_parameter_weights
    
    def run(self):
        with open("hill_climbing_weights.json", "a") as f:
            best_weights = self.optimize()
            best_weights_dict = {key: value for key, value in zip(self._keys, best_weights)}
            dump(best_weights_dict, f, indent=4)
            

if __name__ == "__main__":
    hill_climbing_algorithm = HillClimbingAlgorithm()
    hill_climbing_algorithm.run()
