TestCribbage:
	echo "#!/bin/bash" > TestCribbage
	echo "python3 test_cribbage.py \"\$$@\"" >> TestCribbage
	chmod u+x TestCribbage

GeneticAlgorithm:
	echo "#!/bin/bash" > GeneticAlgorithm
	echo "python3 genetic_optimization_algorithm.py \"\$$@\"" >> GeneticAlgorithm
	chmod u+x GeneticAlgorithm

HillClimbingAlgorithm:
	echo "#!/bin/bash" > HillClimbingAlgorithm
	echo "python3 hill_climbing_optimization_algorithm.py \"\$$@\"" >> HillClimbingAlgorithm
	chmod u+x HillClimbingAlgorithm