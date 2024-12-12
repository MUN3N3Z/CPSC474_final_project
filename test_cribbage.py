import sys
import time
import math
import itertools as it
import argparse
from typing import Tuple, Dict
from json import load 
from policy import CompositePolicy, GreedyThrower, GreedyPegger
from my_policy import StatisticalThrower, RuleBasedPegger
from cribbage import Game, evaluate_policies

ASSESSMENT_GAMES = 1000

def test_policies(game: Game, policy1: CompositePolicy, benchmark: CompositePolicy, run_time:int) -> Tuple[float, float, Dict[int, int]]:
    """ Evaluate two policies against each other in cribbage games for a fixed amount of time
        policy1 - the first policy to evaluate
        benchmark - the BENCHMARK policy to evaluate against
        time - the amount of time to run the evaluation
    """
    match_values = list(it.chain(range(1, 4), range(-3, 0)))
    results = {value: 0 for value in match_values}
    start_time = time.time()
    total_games = 0
    while time.time() - start_time < run_time:
        batch_results = evaluate_policies(game, policy1, benchmark, ASSESSMENT_GAMES)
        total_games += ASSESSMENT_GAMES
        for v in batch_results[3]:
            results[v] += batch_results[3][v]

    sum_squares = sum(results[v] * v * v for v in match_values)
    sum_values = sum(results[v] * v for v in match_values)
    mean = sum_values / total_games
    variance = sum_squares / total_games - math.pow(sum_values / total_games, 2)
    stddev = math.sqrt(variance)
    confidence =  mean - 2 * (stddev / math.sqrt(total_games))

    return mean, confidence, results

def format_output(string: str, mean: float, confidence: float, results_breakdown: Dict[int, int]) -> None:
    """
      Format the output of the test_policies performance comparison
    """
    print(f'{string}: {mean} +/- {confidence}')
    print(results_breakdown)
    print("--------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', type=int, default=3600, help='The amount of time to each evaluation between policies')
    args = parser.parse_args()
    run_time = args.time
   
    with open('genetic_algorithm_weights.json', 'r') as f1:
        # My policy with weights optimized by genetic algorithm VS Greedy policy
        genetic_algorithm_weights = load(f1)
        game = Game()
        policy_with_genetic_weights = CompositePolicy(game, StatisticalThrower(game), RuleBasedPegger(game, genetic_algorithm_weights))
        greedy_benchmark = CompositePolicy(game, GreedyThrower(game), GreedyPegger(game))
        mean, confidence, results_breakdown = test_policies(game, policy_with_genetic_weights, greedy_benchmark, run_time)
        format_output("Genetic Algorithm Weights vs Greedy Policy", mean, confidence, results_breakdown)
        with open('hill_climbing_weights.json', 'r') as f2:
            # My policy with weights optimized by hill climbing algorithm VS Greedy policy
            hill_climbing_algorithm_weights = load(f2)
            game = Game()
            policy_with_hill_climbing_weights = CompositePolicy(game, StatisticalThrower(game), RuleBasedPegger(game, hill_climbing_algorithm_weights))
            mean, confidence, results_breakdown = test_policies(game, policy_with_genetic_weights, greedy_benchmark, run_time)
            format_output("Hill Climbing Weights vs Greedy Policy", mean, confidence, results_breakdown)
            # My policy with weights optimized by genetic algorithm VS My policy with weights optimized by hill climbing (baseline)
            game = Game()
            mean, confidence, results_breakdown = test_policies(game, policy_with_genetic_weights, policy_with_hill_climbing_weights, run_time)
            format_output("Genetic Algorithm Weights vs Hill Climbing Weights (baseline)", mean, confidence, results_breakdown)
            # My policy with weights optimized by hill climbing vd My policy with weights optimized by genetic algorithm (baseline)
            mean, confidence, results_breakdown = test_policies(game, policy_with_hill_climbing_weights, policy_with_genetic_weights, run_time)
            format_output("Hill Climbing Weights vs Genetic Algorithm Weights (baseline)", mean, confidence, results_breakdown)

    
    
