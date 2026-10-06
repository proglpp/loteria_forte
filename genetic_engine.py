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
    def __init__(
        self,
        draws: pd.DataFrame,
        features: pd.DataFrame,
        numbers: List[str] | None = None,
        game_size: int = 20,
    ):
        _ensure_creator()
        self.draws = draws
        self.features = features.set_index("number")
        self.numbers = numbers or [f"{n:02d}" for n in range(100)]
        self.game_size = int(game_size)
        if not 1 <= self.game_size <= len(self.numbers):
            raise ValueError("game_size must be within the provided number universe")
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._make_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("evaluate", self._evaluate)
        self.criteria = "score_total"
        random.seed(42)

    def _make_individual(self) -> List[str]:
        return random.sample(self.numbers, self.game_size)

    def _repair_individual(self, individual) -> None:
        unique = list(dict.fromkeys(individual))[: self.game_size]
        available = [number for number in self.numbers if number not in unique]
        if len(unique) < self.game_size:
            unique.extend(random.sample(available, self.game_size - len(unique)))
        individual[:] = unique

    def _mutate(self, individual):
        idx = random.randrange(len(individual))
        candidate = random.choice([number for number in self.numbers if number not in individual or number == individual[idx]])
        individual[idx] = candidate
        return individual,

    def _evaluate(self, individual: List[str]):
        unique_set = set(individual)
        if len(unique_set) < self.game_size:
            return 0.0, 0.0, 100.0
        score = self.features.loc[list(unique_set), "score_total"].sum()
        diversity = len(unique_set)
        coverage = len(unique_set) / float(self.game_size)
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
        for individual in population:
            individual.fitness.values = self.toolbox.evaluate(individual)
        population = tools.selNSGA2(population, len(population))
        for generation in range(generations):
            offspring = tools.selTournament(population, len(population), tournsize=3)
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.9:
                    self.toolbox.mate(child1, child2)
                    self._repair_individual(child1)
                    self._repair_individual(child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mutant)
                    self._repair_individual(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = self.toolbox.select(population + offspring, population_size)
        results = []
        for individual in population[:20]:
            unique_numbers = sorted(set(individual))[: self.game_size]
            results.append({
                "game": unique_numbers,
                "score_total": sum(self.features.loc[unique_numbers, "score_total"]),
                "unique_count": len(unique_numbers),
                "fitness_criteria": fitness_criteria,
            })
        return pd.DataFrame(results)
