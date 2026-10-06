from math import comb

import pytest

from lotofacil_dashboard import (
    NUMBERS,
    build_lotofacil_movement_table,
    build_lotofacil_ranking,
    generate_lotofacil_games,
    load_lotofacil_history,
    lotofacil_ticket_odds,
)


def test_official_history_uses_lotofacil_rules_and_latest_draw():
    draws = load_lotofacil_history()

    assert len(draws) == 3797
    assert int(draws.iloc[-1]["Concurso"]) == 3797
    assert draws.iloc[-1]["Data Sorteio"].strftime("%d/%m/%Y") == "05/10/2026"
    assert draws.iloc[-1][[f"Bola{index}" for index in range(1, 16)]].nunique() == 15
    assert draws[[f"Bola{index}" for index in range(1, 16)]].apply(
        lambda column: column.map(lambda number: 1 <= int(number) <= 25)
    ).values.all()


def test_simple_ticket_top_prize_probability_matches_official_odds():
    odds = lotofacil_ticket_odds(15)

    assert odds[15] == pytest.approx(1 / comb(25, 15))
    assert odds[11] >= odds[12] >= odds[13] >= odds[14] >= odds[15]


def test_generator_creates_distinct_tickets_with_fixed_numbers():
    ranking = build_lotofacil_ranking(load_lotofacil_history())
    games = generate_lotofacil_games(
        ranking,
        ticket_size=17,
        game_count=5,
        method="Score equilibrado",
        selected=["01", "25"],
        seed=7,
    )

    assert len(games) == 5
    assert len({tuple(game) for game in games}) == 5
    assert all(len(game) == 17 and set(game).issubset(NUMBERS) for game in games)
    assert all({"01", "25"}.issubset(game) for game in games)


@pytest.mark.parametrize("ticket_size", [15, 20])
def test_pattern_filtered_generator_returns_valid_lotofacil_tickets(ticket_size):
    draws = load_lotofacil_history()
    ranking = build_lotofacil_ranking(draws)
    games = generate_lotofacil_games(
        ranking,
        ticket_size=ticket_size,
        game_count=3,
        method="Score equilibrado",
        seed=11,
        draws=draws,
        patterns=["Pares e ímpares", "Faixa de soma", "Linhas e colunas", "Repetidas do último"],
    )

    assert len(games) == 3
    assert all(len(game) == ticket_size for game in games)
    assert all(len(set(game)) == ticket_size and set(game).issubset(NUMBERS) for game in games)


def test_movement_table_has_draw_rows_metrics_and_color_statuses():
    draws = load_lotofacil_history()
    table = build_lotofacil_movement_table(draws, rows=8, strength_window=10)

    assert list(table.columns) == ["Concurso", *NUMBERS]
    assert len(table) == 12
    assert {"FREQ", "ATRASO", "TENDÊNCIA", "FORÇA"}.issubset(set(table["Concurso"]))
    assert set(table.attrs["status_map"]) == set(NUMBERS)
    assert set(table.attrs["status_map"].values()).issubset({"FORTE", "BOM", "OBS", "FRACO"})


@pytest.mark.parametrize("ticket_size", [14, 21])
def test_odds_reject_ticket_sizes_outside_lotofacil_rules(ticket_size):
    with pytest.raises(ValueError, match="15 a 20"):
        lotofacil_ticket_odds(ticket_size)