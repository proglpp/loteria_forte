"""Compact visual UI components for Lotomania game generation."""

from html import escape
from typing import Dict, List, Set

import streamlit as st


def _normalize_number_set(values) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = values.replace(",", " ").replace(";", " ").split()

    normalized = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= number <= 99:
            normalized.add(f"{number:02d}")
    return normalized


def _number_palette(status: str, selected: bool = False, fixed: bool = False) -> tuple[str, str, str]:
    palettes = {
        "FORTE": ("#009966", "#ffffff", "#007a52"),
        "BOM": ("#55b7ff", "#07121f", "#2185c7"),
        "OBS": ("#d6b4ec", "#2b0f3f", "#9b5ec4"),
        "FRACO": ("#e85d75", "#ffffff", "#b9354d"),
    }
    background, color, border = palettes.get(status, ("#f7f2fb", "#2b0f3f", "#c8a4dc"))
    if selected:
        background, color, border = "#ffd166", "#351400", "#d97706"
    if fixed:
        background, color, border = "#5b159f", "#ffffff", "#3f0f73"
    return background, color, border


def _inject_generator_css() -> None:
    st.markdown(
        """
        <style>
        .lotomania-board {
            display: inline-grid;
            grid-template-columns: repeat(10, 29px);
            gap: 4px;
            padding: 12px;
            border: 1px solid #b0189d;
            border-radius: 8px;
            background: #fff7b8;
            box-shadow: 0 5px 16px rgba(91, 21, 159, 0.14);
        }
        .lotomania-board span {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 27px;
            height: 24px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 800;
            line-height: 1;
        }
        .lotomania-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 4px;
            font-size: 12px;
        }
        .lotomania-legend span {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .lotomania-legend i {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }
        .game-pill-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 7px 0 10px;
        }
        .game-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 28px;
            border-radius: 999px;
            border: 1px solid #de6b2f;
            background: #ffd6bd;
            color: #2e1608;
            font-size: 13px;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_compact_number_board(selected: Set[str], fixed: Set[str], status_map: Dict[str, str] | None = None) -> None:
    status_map = status_map or {}
    cells = []
    ordered_numbers = [f"{number:02d}" for number in range(1, 100)] + ["00"]
    for number in ordered_numbers:
        status = status_map.get(number, "OBS")
        background, color, border = _number_palette(status, selected=number in selected, fixed=number in fixed)
        title = f"{number} - {status}"
        cells.append(
            f'<span title="{escape(title)}" style="background:{background};color:{color};'
            f'border:1px solid {border};">{number}</span>'
        )

    _inject_generator_css()
    st.markdown(f'<div class="lotomania-board">{"".join(cells)}</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="lotomania-legend">
            <span><i style="background:#009966"></i>Forte</span>
            <span><i style="background:#55b7ff"></i>Bom</span>
            <span><i style="background:#d6b4ec"></i>Obs</span>
            <span><i style="background:#e85d75"></i>Fraco</span>
            <span><i style="background:#ffd166"></i>Selecionado</span>
            <span><i style="background:#5b159f"></i>Fixo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _selection_stats(numbers: List[str]) -> dict:
    ints = [int(number) for number in numbers]
    return {
        "quantidade": len(numbers),
        "pares": sum(1 for number in ints if number % 2 == 0),
        "impares": sum(1 for number in ints if number % 2 != 0),
        "soma": sum(ints),
        "media": round(sum(ints) / len(ints), 2) if ints else 0,
    }


def render_game_selector_panel(status_map: Dict[str, str] | None = None) -> dict:
    if "ui_selected_numbers" not in st.session_state:
        st.session_state.ui_selected_numbers = set()
    if "ui_fixed_numbers" not in st.session_state:
        st.session_state.ui_fixed_numbers = set()

    selected_set = _normalize_number_set(st.session_state.ui_selected_numbers)
    fixed_set = _normalize_number_set(st.session_state.ui_fixed_numbers) & selected_set
    all_numbers = [f"{number:02d}" for number in range(1, 100)] + ["00"]

    st.markdown("#### Gerador de Jogos para a Lotomania")
    board_col, control_col = st.columns([1.05, 1.0], gap="large")

    with board_col:
        render_compact_number_board(selected_set, fixed_set, status_map=status_map)
        stats = _selection_stats(sorted(selected_set))
        st.caption(
            f"Selecionados:{stats['quantidade']} | Par:{stats['pares']} "
            f"Impar:{stats['impares']} | Soma:{stats['soma']}"
        )

    with control_col:
        st.markdown("#### Filtros e Regras")
        rule_options = [
            "Numeros pares e impares",
            "Soma dos numeros",
            "Dezenas por linha",
            "Dezenas por coluna",
            "Repetidas no concurso anterior",
            "Sequencia grande de numeros",
            "Sequencia grande de saltos",
            "Numeros de Fibonacci",
            "Numeros primos",
            "Pesos inteligentes",
            "Aposta espelho",
        ]
        enabled_rules = []
        for index, label in enumerate(rule_options):
            if st.checkbox(label, value=index < 9, key=f"generator_rule_{index}"):
                enabled_rules.append(label)

        selected = st.multiselect(
            "Numeros selecionados",
            all_numbers,
            default=sorted(selected_set),
            max_selections=50,
            key="compact_selected_numbers",
        )
        fixed = st.multiselect(
            "Fixar numeros",
            all_numbers,
            default=sorted(fixed_set),
            max_selections=50,
            key="compact_fixed_numbers",
        )
        if st.button("Novo", key="compact_clear_selection"):
            st.session_state.ui_selected_numbers = set()
            st.session_state.ui_fixed_numbers = set()
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

    selected = sorted(_normalize_number_set(selected))
    fixed = sorted(_normalize_number_set(fixed) & set(selected))
    st.session_state.ui_selected_numbers = set(selected)
    st.session_state.ui_fixed_numbers = set(fixed)

    game_config = {"numbers": selected, "fixed": fixed, "rules": enabled_rules}
    game_config.update(_selection_stats(selected))
    return game_config


def render_generated_games_gallery(games: List[List[str]]) -> None:
    st.markdown("### Numeros Gerados")
    if not games:
        st.info("Nenhum jogo gerado ainda.")
        return

    for index, game in enumerate(games, start=1):
        game_numbers = sorted([f"{int(number):02d}" for number in game])
        game_ints = [int(number) for number in game_numbers]
        game_str = ", ".join(game_numbers)
        pills = "".join(f'<span class="game-pill">{escape(number)}</span>' for number in game_numbers)
        st.markdown(f"**Jogo:{index:02d}**")
        st.markdown(f'<div class="game-pill-wrap">{pills}</div>', unsafe_allow_html=True)
        st.caption(
            f"Par:{sum(1 for number in game_ints if number % 2 == 0)} | "
            f"Impar:{sum(1 for number in game_ints if number % 2 != 0)} | Soma:{sum(game_ints)}"
        )
        st.download_button(
            label=f"Baixar jogo {index}",
            data=game_str,
            file_name=f"jogo_{index}.txt",
            mime="text/plain",
            key=f"download_game_{index}",
        )
