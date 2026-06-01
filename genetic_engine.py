import logging
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from deap import base, creator, tools

logger = logging.getLogger(__name__)


def _ensure_creator():
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)


class GeneticGameOptimizer:
    def __init__(self, draws: pd.DataFrame, features: pd.DataFrame):
        _ensure_creator()
        self.draws = draws
        self.features = features.set_index("number")
        self.numbers = [f"{n:02d}" for n in range(100)]
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_number", lambda: random.choice(self.numbers))
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self._unique_number, 20)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self._evaluate)
        self.criteria = "score_total"
        random.seed(42)

    def _unique_number(self) -> str:
        return random.choice(self.numbers)

    def _mutate(self, individual):
        idx = random.randrange(len(individual))
        candidate = random.choice(self.numbers)
        individual[idx] = candidate
        return individual,

    def _evaluate(self, individual: List[str]):
        unique_set = set(individual)
        if len(unique_set) < 20:
            return 0.0, 0.0, 100.0
        score = self.features.loc[list(unique_set), "score_total"].sum()
        diversity = len(unique_set)
        coverage = len(unique_set) / 20.0
        if self.criteria == "score_total":
            return float(score), float(diversity), -float(coverage)
        if self.criteria == "diversity":
            return float(score * 0.1), float(diversity), -float(coverage)
        if self.criteria == "coverage":
            return float(score * 0.1), float(coverage * 100.0), -float(diversity)
        return float(score * 0.5), float(diversity), -float(coverage)

    def run_evolutionary_search(
        self,
        population_size: int = 50,
        generations: int = 20,
        fitness_criteria: str = "score_total",
    ) -> pd.DataFrame:
        logger.info("Running evolutionary game optimizer with criteria %s", fitness_criteria)
        self.criteria = fitness_criteria
        population = self.toolbox.population(n=population_size)
        algorithms = tools
        population = algorithms.selNSGA2(population, len(population))
        for generation in range(generations):
            offspring = tools.selTournament(population, len(population), tournsize=3)
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = self.toolbox.select(population + offspring, population_size)
        results = []
        for individual in population[:20]:
            unique_numbers = sorted(set(individual))[:20]
            results.append({
                "game": unique_numbers,
                "score_total": sum(self.features.loc[unique_numbers, "score_total"]),
                "unique_count": len(unique_numbers),
                "fitness_criteria": fitness_criteria,
            })
        return pd.DataFrame(results)
