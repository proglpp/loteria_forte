import logging
from typing import Dict

import networkx as nx
import pandas as pd

from config import DRAW_COLUMNS, NUMBER_RANGE

logger = logging.getLogger(__name__)


def build_graph_metrics(draws: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building graph metrics")
    graph = nx.Graph()
    numbers = [f"{n:02d}" for n in NUMBER_RANGE]
    graph.add_nodes_from(numbers)

    for _, row in draws.iterrows():
        draw_numbers = [str(row[col]) for col in DRAW_COLUMNS]
        for i, number_a in enumerate(draw_numbers):
            for number_b in draw_numbers[i + 1 :]:
                if graph.has_edge(number_a, number_b):
                    graph[number_a][number_b]["weight"] += 1
                else:
                    graph.add_edge(number_a, number_b, weight=1)

    pagerank = nx.pagerank(graph, weight="weight")
    degree = dict(graph.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    closeness = nx.closeness_centrality(graph, distance=None)
    eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=500, tol=1e-06)
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    community_map = {}
    for community_index, community_nodes in enumerate(communities, start=1):
        for number in community_nodes:
            community_map[number] = f"C{community_index}"

    metrics = []
    for number in numbers:
        community_value = community_map.get(number, "C0")
        try:
            community_code = int(community_value.lstrip("C"))
        except ValueError:
            community_code = 0
        metrics.append(
            {
                "number": number,
                "pagerank": pagerank.get(number, 0.0),
                "degree": degree.get(number, 0.0),
                "betweenness": betweenness.get(number, 0.0),
                "closeness": closeness.get(number, 0.0),
                "eigenvector": eigenvector.get(number, 0.0),
                "community": community_value,
                "community_code": community_code,
            }
        )
    return pd.DataFrame(metrics)
