import pygad
import numpy as np
import itertools as it
from cribbage import Game, evaluate_policies
from typing import List, Dict, Tuple
from deck import Card
from pegging import Pegging
from my_policy import StatisticalThrower, RuleBasedPegger
from policy import CompositePolicy, GreedyThrower, GreedyPegger, CribbagePolicy

ASSESSMENT_GAMES = 1000

class GeneticAlgorithmPolicyWithStatisticalThrower(CribbagePolicy):
    def __init__(self, game: Game, card_weights: Dict[str, float]):
        super().__init__(game)
        self._policy = CompositePolicy(game, StatisticalThrower(game), RuleBasedPegger(game, card_weights))
    
    def keep(self, hand: List[Card], scores: Tuple[int], am_dealer: bool):
        return self._policy.keep(hand, scores, am_dealer)
    
    def peg(self, cards: List[Card], history: Pegging, turn: Card, scores: Tuple[int], am_dealer: bool):
        return self._policy.peg(cards, history, turn, scores, am_dealer)
        
class PeggingGeneticAlgorithm:
    def __init__(self):
        # Set genetic algorithm parameters
        self._num_weights = 8
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
            num_genes=self._num_weights,
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


    def _fitness(self, pygad_instance, solution: np.ndarray[np.float16], solution_idx: int):
        """
            - Run a cribbage agent using the weights from the solution in its rule-based pegging strategy and
            compare its performance against a baseline greedy pegger
            - Both agents will use the same statistical throwing strategy
        """
        keys = [
            'score_points', 'lead_low_card', 'lead_sum_to_15', 
            'closest_to_31', 'save_ace', 'play_ace', 
            'penalize_5', 'illegal_play'
        ]
        card_weights = {key: value for key, value in zip(keys, solution)}
        game = Game()
        benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        genetic_algorithm_policy = GeneticAlgorithmPolicyWithStatisticalThrower(game, card_weights)
        results = evaluate_policies(game, genetic_algorithm_policy, benchmark, ASSESSMENT_GAMES)
        points = 0
        for match_value, number_of_games in results[3].items():
            points += match_value * match_value * number_of_games
        return points / ASSESSMENT_GAMES
    
    def run(self):
        self.ga_instance.run()
        solution, solution_fitness, _ = self.ga_instance.best_solution()
        with open("genetic_algorithm_best_weights2.txt", "a") as f:
            f.write(f"Best weights: {solution}\n")
            f.write(f"Best score: {solution_fitness}\n")
            f.write("\n")

if __name__ == "__main__":
   PeggingGeneticAlgorithm().run()
         
