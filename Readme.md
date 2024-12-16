---
CPSC 474 Final Project
---

#### <u> Investigating the how effective pegging parameters optimized by a Genetic Algorithm are when used in a rule-based Cribbage pegging agent compared to those optimized by  a Hill Climbing Algorithm. </u>

#### Results
The genetic algorithm ran for about 36 hours while the hill climbing algorithm ran for about 24 hours. I let the testing script - `test_cribbage.py` - run for about 24 hours to get a good sample size of games played. The results are as follows:
- Policy with genetically-optimized pegging parameters averaged **0.041** match points when playing against a policy with a greedy pegging strategy.
- Policy with parameters optimized by a hill climbing algorithm averaged **0.063** match points when playing against a policy with a greedy pegging strategy.
- Policy with genetically-optimized pegging parameters averaged **-0.028** match points when playing against a policy with parameters optimized by a hill climbing algorithm.
- Policy with parameters optimized by a hill climbing algorithm averaged **0.026** match points when playing against a policy with parameters optimized by a genetic algorithm.

**Note:** The throwing strategy for all policies was the same. The throwing strategy was dictated by a statistical sampling method that I implemented to create my Schell's table. 

#### Conclusion
- The hill climbing algorithm was able to find better parameters than the genetic algorithm. I can attribute this to the amount of randomness introduced in the hill climbing algorithm by reducing the number of evaluation games to 50 compared to 1000 for the genetic algorithm. This randomness allowed the hill climbing algorithm to explore more of the search space and find better parameters.
- Given more experimentation time, I would like to see how the genetic algorithm would perform with a smaller population size and a smaller number of generations. 
- My Schell's table could be another source of inaccuracy in my results. I would have liked to experiment using the official Schell's table to see how my results would change.

#### Instructions
- To run the genetic algorithm, run `./GeneticAlgorithm` in the terminal. This will run the genetic algorithm and save the best parameters to `genetic_algorithm_weights.json`.
- To run the hill climbing algorithm, run `./HillClimbingAlgorithm < --time x >` where `x` is the amount of time in **hours** you would like the algorithm to run. This will run the hill climbing algorithm and save the best parameters to `hill_climbing_algorithm_weights.json`.
- To run the testing script, run `./TestCribbage < --time x >` where `x` is the amount of time in **hours** you would like the test script to run. The default time is **5 minutes**. This will run the testing script and output the results to the terminal.