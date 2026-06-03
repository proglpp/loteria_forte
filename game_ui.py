"""Visual UI components for game selection and generation."""

from html import escape
from typing import List, Set, Tuple

import pandas as pd
import streamlit as st


def _normalize_number_set(values) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]

    normalized = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= number <= 99:
            normalized.add(f"{number:02d}")
    return normalized


def render_number_grid(
    width: int = 10,
    selected: Set[str] = None,
    fixed: Set[str] = None,
    columns_count: int = 10,
) -> Tuple[List[str], List[str]]:
    """
    Render an interactive grid of numbers (00-99) for selection with fixed numbers support.
    
    Args:
        width: Total width of the container (not used, for future layout)
        selected: Initial set of selected numbers as strings
        fixed: Initial set of fixed numbers as strings
        columns_count: Number of columns in the grid
    
    Returns:
        Tuple: (sorted list of selected numbers, sorted list of fixed numbers)
    """
    if selected is None:
        selected = set()
    if fixed is None:
        fixed = set()
    
    if "ui_selected_numbers" not in st.session_state:
        st.session_state.ui_selected_numbers = set(selected)
    if "ui_fixed_numbers" not in st.session_state:
        st.session_state.ui_fixed_numbers = set(fixed)

    st.session_state.ui_selected_numbers = _normalize_number_set(st.session_state.ui_selected_numbers)
    st.session_state.ui_fixed_numbers = _normalize_number_set(st.session_state.ui_fixed_numbers)
    
    st.markdown("**Seletor de Números (00-99)**")
    
    # Control buttons
    col_controls = st.columns(3)
    with col_controls[0]:
        toggle_fix = st.checkbox("🔒 Fixar selecionados", value=False, key="toggle_fix_numbers")
    with col_controls[1]:
        if st.button("🗑️ Limpar", key="btn_clear_selection", use_container_width=True):
            st.session_state.ui_selected_numbers = set(st.session_state.ui_fixed_numbers)
            st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()
    with col_controls[2]:
        st.write(f"**Selecionados:** {len(st.session_state.ui_selected_numbers)}")
    
    # Apply toggle fix logic
    if toggle_fix:
        # Move selected to fixed
        st.session_state.ui_fixed_numbers.update(st.session_state.ui_selected_numbers)
    else:
        # Remove from fixed only those that aren't currently selected
        st.session_state.ui_fixed_numbers = st.session_state.ui_fixed_numbers & st.session_state.ui_selected_numbers
    
    # Grid
    cols = st.columns(columns_count)
    
    for num in range(100):
        col = cols[num % columns_count]
        num_str = f"{num:02d}"
        button_key = f"num_{num_str}"
        
        is_selected = num_str in st.session_state.ui_selected_numbers
        is_fixed = num_str in st.session_state.ui_fixed_numbers
        
        # Determine button label and styling
        if is_fixed:
            button_label = f"🔒{num_str}"
        elif is_selected:
            button_label = f"✓{num_str}"
        else:
            button_label = f"{num_str}"
        
        if col.button(button_label, key=button_key, use_container_width=True):
            if is_selected:
                # Only allow removal if not fixed
                if not is_fixed:
                    st.session_state.ui_selected_numbers.discard(num_str)
            else:
                # Add to selection if not at limit
                if len(st.session_state.ui_selected_numbers) < 50:
                    st.session_state.ui_selected_numbers.add(num_str)
    
    return sorted(list(st.session_state.ui_selected_numbers)), sorted(list(st.session_state.ui_fixed_numbers))



def render_game_parameters(game_data: dict = None) -> dict:
    """
    Render game parameter controls and analysis (compact version).
    
    Args:
        game_data: Optional dict with 'numbers' list
    
    Returns:
        Dict with game parameters and statistics
    """
    st.markdown("**Estatísticas**")
    
    if game_data is None:
        game_data = {}
    
    numbers = game_data.get("numbers", [])
    if isinstance(numbers, str):
        numbers = numbers.split(",")
    
    # Convert to integers for analysis
    try:
        num_ints = [int(n) for n in numbers]
    except (ValueError, TypeError):
        num_ints = []
    
    evens = sum(1 for n in num_ints if n % 2 == 0)
    odds = sum(1 for n in num_ints if n % 2 != 0)
    total_sum = sum(num_ints)
    
    # Compact display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Qtd", len(numbers), delta=None)
        st.metric("Pares", evens, delta=None)
    
    with col2:
        st.metric("Ímpares", odds, delta=None)
        st.metric("Soma", total_sum, delta=None)
    
    with col3:
        st.metric("Média", round(total_sum / len(numbers), 1) if numbers else 0, delta=None)
        st.metric("Min/Max", f"{min(num_ints)}/{max(num_ints)}" if num_ints else "0/0", delta=None)
    
    # Display numbers (compact)
    if numbers:
        numbers_text = ", ".join(sorted([f"{int(n):02d}" for n in numbers]))
        st.caption(f"Números: {numbers_text}")
    
    return {
        "quantidade": len(numbers),
        "pares": evens,
        "impares": odds,
        "soma": total_sum,
        "media": round(total_sum / len(numbers), 2) if numbers else 0,
        "minimo": min(num_ints) if num_ints else 0,
        "maximo": max(num_ints) if num_ints else 0,
    }




def render_game_simulations(game_data: dict = None) -> None:
    """
    Render a table showing prize simulations based on game numbers.
    
    Args:
        game_data: Dict with 'numbers' list
    """
    if game_data is None or not game_data.get("numbers"):
        return
    
    with st.expander("📊 Simulador de Prêmios", expanded=False):
        # Simulate prize tiers
        simulation_data = {
            "Faixa": ["15 pts", "16 pts", "17 pts", "18 pts", "19 pts", "20 pts"],
            "Qtd": [0, 0, 0, 0, 0, 0],
        }
        
        df_sim = pd.DataFrame(simulation_data)
        st.dataframe(df_sim, use_container_width=True, hide_index=True)




def render_game_selector_panel() -> dict:
    """
    Render complete game selection panel with all components.
    
    Returns:
        Dict with game configuration
    """
    # Initialize session state for selected numbers
    if "ui_selected_numbers" not in st.session_state:
        st.session_state.ui_selected_numbers = set()
    if "ui_fixed_numbers" not in st.session_state:
        st.session_state.ui_fixed_numbers = set()
    
    # Number grid
    selected, fixed = render_number_grid(
        selected=st.session_state.ui_selected_numbers,
        fixed=st.session_state.ui_fixed_numbers
    )
    st.session_state.ui_selected_numbers = set(selected)
    st.session_state.ui_fixed_numbers = set(fixed)
    
    # Game parameters
    game_config = {
        "numbers": selected,
        "fixed": fixed,
    }
    params = render_game_parameters(game_config)
    game_config.update(params)
    
    # Simulations
    render_game_simulations(game_config)
    
    return game_config



def render_generated_games_gallery(games: List[List[str]]) -> None:
    """
    Render gallery of generated games.
    
    Args:
        games: List of games, each game is a list of number strings
    """
    st.markdown("### Jogos Gerados")
    st.markdown(
        """
        <style>
            .generated-game-numbers {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                width: 100%;
                min-width: 0;
                padding: 0.75rem 1rem;
                border-radius: 0.5rem;
                background: rgba(49, 51, 63, 0.38);
                border: 1px solid rgba(250, 250, 250, 0.08);
            }

            .generated-game-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 2.25rem;
                height: 1.75rem;
                padding: 0 0.35rem;
                border-radius: 0.35rem;
                background: rgba(250, 250, 250, 0.08);
                color: inherit;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
                font-size: 0.9rem;
                font-weight: 700;
                line-height: 1;
                white-space: nowrap;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    if not games:
        st.info("Nenhum jogo gerado ainda.")
        return
    
    for idx, game in enumerate(games, start=1):
        with st.expander(f"🎮 Jogo {idx} - {len(game)} números", expanded=(idx == 1)):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Display numbers
                game_numbers = sorted([f"{int(n):02d}" for n in game])
                game_str = ", ".join(game_numbers)
                number_chips = "".join(
                    f'<span class="generated-game-number">{escape(number)}</span>'
                    for number in game_numbers
                )
                st.markdown(
                    f"""
                    <div class="generated-game-numbers" aria-label="Números do jogo {idx}">
                        {number_chips}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                # Quick stats
                game_ints = [int(n) for n in game]
                st.metric("Soma", sum(game_ints))
            
            with col3:
                # Download button
                st.download_button(
                    label=f"⬇️ Baixar",
                    data=game_str,
                    file_name=f"jogo_{idx}.txt",
                    mime="text/plain",
                    key=f"download_game_{idx}",
                    use_container_width=True,
                )

