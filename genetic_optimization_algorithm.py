import pygad
import numpy as np
import itertools as it
from cribbage import Game, evaluate_policies
from my_policy import StatisticalThrower, RuleBasedPegger
from policy import CompositePolicy, GreedyThrower, GreedyPegger
from json import dump

ASSESSMENT_GAMES = 1000
        
class PeggingGeneticAlgorithm:
    def __init__(self):
        # Set genetic algorithm parameters
        self._gene_space = [
            {'low': 0, 'high': 5},  # score_points
            {'low': 0, 'high': 2},  # lead_low_card
            {'low': 0, 'high': 2},  # lead_sum_to_15
            {'low': 0, 'high': 2},  # closest_to_31
            {'low': -2, 'high': 0},  # save_ace
            {'low': 0, 'high': 2},  # play_ace
            {'low': -2, 'high': 0},  # penalize_5
            {'low': -5, 'high': 0},  # illegal_play
        ]
        self.ga_instance = pygad.GA(
            num_generations=20,
            num_parents_mating=3,
            fitness_func=self._fitness,
            sol_per_pop=20,
            num_genes=8,
            gene_space=self._gene_space,
            parent_selection_type="tournament",
            crossover_type="two_points",
            mutation_percent_genes=20,
            keep_parents=1,
            crossover_probability=0.4,
            mutation_probability=0.1,
            mutation_type="random",
            save_best_solutions=False,
        )
        self._match_values = list(it.chain(range(1, 4), range(-3, 0)))
        self._keys = [
            'score_points', 'lead_low_card', 'lead_sum_to_15', 
            'closest_to_31', 'save_ace', 'play_ace', 
            'penalize_5', 'illegal_play'
        ]

    def _fitness(self, pygad_instance, solution: np.ndarray[np.float16], solution_idx: int):
        """
            - Run a cribbage agent using the weights from the solution in its rule-based pegging strategy and
            compare its performance against a baseline greedy pegger
            - Both agents will use the same statistical throwing strategy
        """
        
        card_weights = {key: value for key, value in zip(self._keys, solution)}
        game = Game()
        benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        policy_with_genetic_weights = CompositePolicy(game, StatisticalThrower(game), RuleBasedPegger(game, card_weights))
        results = evaluate_policies(game, policy_with_genetic_weights, benchmark, ASSESSMENT_GAMES)
        points = 0
        for match_value, number_of_games in results[3].items():
            points += match_value * match_value * number_of_games
        return points / ASSESSMENT_GAMES
    
    def run(self):
        self.ga_instance.run()
        solution, _, _ = self.ga_instance.best_solution()
        with open("genetic_algorithm_weights.json", "a") as f:
            best_weights_dict = {key: value for key, value in zip(self._keys, solution)}
            dump(best_weights_dict, f, indent=4)

if __name__ == "__main__":
   PeggingGeneticAlgorithm().run()
         
