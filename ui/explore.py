"""Tab 2's "Explore the problem" panel: parameter widgets on the left, the explorer page on the right.

The widgets rebuild `ShuttleParams`; `rl.dp.explore.explore` runs the real solvers on them (cached per
parameter set; milliseconds) and `render_html` embeds the result in rl/dp/explore.html, which the panel
shows in an iframe. The same page can be written to disk with `python run.py dp --explorer FILE`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict

import streamlit as st
import streamlit.components.v1 as components

from rl.dp.explore import explore, render_html
from rl.dp.shuttle import HOLDING_CHARGED, ShuttleParams

HEIGHT = 1330  # iframe height; the page scrolls inside it if a parameter set needs more

FIELDS = {
    "capacity": 15, "dispatch_cost": 100.0, "holding_cost": 2.0, "arrivals": "1, 2, 3, 4, 5", "arrival_probs": "",
    "max_customers": 200, "gamma": 0.95, "horizon": 500, "max_dispatch": 1, "holding_charged": "after_dispatch",
    "tol": "1e-8",
}
PRESETS = {
    "Handout parameters": {},
    "Deterministic arrivals (A_t = 3)": {"arrivals": "3"},
    "Non-uniform arrivals (1 with probability 0.8, 5 with 0.2)": {"arrivals": "1, 5", "arrival_probs": "0.8, 0.2"},
    "Myopic agent (γ = 0)": {"gamma": 0.0},
    "Patient agent (γ = 0.99)": {"gamma": 0.99},
    "No discounting over T = 20 periods (γ = 1)": {"gamma": 1.0, "horizon": 20},
    "Tiny station (M = 20, K = 5)": {"max_customers": 20, "capacity": 5},
    "Expensive waiting (c_h = 20)": {"holding_cost": 20.0},
    "Cheap shuttle (c_f = 10)": {"dispatch_cost": 10.0},
    "Free waiting (c_h = 0)": {"holding_cost": 0.0},
    "Short horizon (T = 5)": {"horizon": 5},
    "Up to two shuttles per period": {"max_dispatch": 2},
    "Holding cost charged before the dispatch": {"holding_charged": "before_dispatch"},
}
TOLERANCES = ["1e-4", "1e-6", "1e-8", "1e-10"]


def _key(field: str) -> str:
    return f"xp_{field}"


def _init_state() -> None:
    st.session_state.setdefault("xp_preset", next(iter(PRESETS)))
    for field, value in FIELDS.items():
        st.session_state.setdefault(_key(field), value)


def _apply_preset() -> None:
    values = {**FIELDS, **PRESETS[st.session_state["xp_preset"]]}
    for field, value in values.items():
        st.session_state[_key(field)] = value


def _reset() -> None:
    st.session_state["xp_preset"] = next(iter(PRESETS))
    _apply_preset()


def _numbers(text: str, kind):
    return tuple(kind(x) for x in re.split(r"[\s,;]+", text.strip()) if x)


def params_from_state() -> ShuttleParams:
    """ShuttleParams from the widgets; ValueError with a readable message when they do not make sense."""
    get = lambda field: st.session_state[_key(field)]  # noqa: E731
    try:
        arrivals = _numbers(get("arrivals"), int)
    except ValueError:
        raise ValueError("arrivals must be whole numbers separated by commas, e.g. 1, 2, 3") from None
    if not arrivals:
        raise ValueError("list at least one possible number of arrivals")
    probs_text = get("arrival_probs").strip()
    try:
        probs = _numbers(probs_text, float) if probs_text else None
    except ValueError:
        raise ValueError("arrival probabilities must be numbers separated by commas, e.g. 0.8, 0.2") from None
    return ShuttleParams(
        capacity=int(get("capacity")),
        dispatch_cost=float(get("dispatch_cost")),
        holding_cost=float(get("holding_cost")),
        arrivals=arrivals,
        arrival_probs=probs,
        max_customers=int(get("max_customers")),
        gamma=float(get("gamma")),
        horizon=int(get("horizon")),
        max_dispatch=int(get("max_dispatch")),
        holding_charged=get("holding_charged"),
    )


@st.cache_data(show_spinner=False, max_entries=64)
def _explore_cached(params_json: str, tol: float) -> dict:
    return explore(ShuttleParams(**json.loads(params_json)), tol=tol)


def _theme() -> str | None:
    try:
        kind = st.context.theme.type
    except Exception:  # noqa: BLE001 - older Streamlit without st.context.theme
        return None
    return kind if kind in ("light", "dark") else None


def parameter_widgets() -> None:
    st.selectbox("Preset", list(PRESETS), key="xp_preset", on_change=_apply_preset,
                 help="Each preset changes one thing about the handout's problem; edit any field afterwards.")
    c1, c2 = st.columns(2)
    c1.number_input("Capacity K", min_value=0, max_value=100_000, step=1, key=_key("capacity"), help="customers one shuttle carries")
    c2.number_input("Station limit M", min_value=1, max_value=5000, step=1, key=_key("max_customers"),
                    help="states are 0 … M customers waiting; arrivals beyond M are turned away")
    c1.number_input("Dispatch cost c_f", min_value=0.0, step=10.0, format="%g", key=_key("dispatch_cost"))
    c2.number_input("Holding cost c_h", min_value=0.0, step=1.0, format="%g", key=_key("holding_cost"), help="per customer waiting, per period")
    st.text_input("Arrivals A_t: possible values", key=_key("arrivals"), help="comma-separated whole numbers")
    st.text_input("Arrival probabilities (blank = uniform)", key=_key("arrival_probs"), help="one per value, summing to 1")
    c3, c4 = st.columns(2)
    c3.number_input("Discount γ", min_value=0.0, max_value=1.0, step=0.01, format="%.3f", key=_key("gamma"),
                    help="γ = 1 disables value and policy iteration (no infinite-horizon solution); enumeration still works")
    c4.number_input("Horizon T", min_value=0, max_value=100_000, step=1, key=_key("horizon"), help="periods for enumeration (part a)")
    c3.number_input("Shuttles per period", min_value=1, max_value=5, step=1, key=_key("max_dispatch"), help="actions 0 … this many dispatches")
    c4.selectbox("Value iteration tol", TOLERANCES, key=_key("tol"))
    st.selectbox("Holding cost charged", HOLDING_CHARGED, format_func=lambda s: s.replace("_", " "), key=_key("holding_charged"),
                 help="which customers pay c_h: those left after the dispatch (handout), everyone present before it, or everyone there at the end of the period")
    st.button("Back to the handout's parameters", on_click=_reset, key="xp_reset")


def explore_panel() -> None:
    """The whole panel: widgets, solve, embed."""
    _init_state()
    with st.expander("Explore the problem", expanded=True, icon=":material/explore:"):
        st.markdown(
            "Change the problem on the left and the three solvers rerun on it immediately. The page on the right "
            "shows the two stations, replays the first 10 periods of any policy (step, play, or take a different "
            "action yourself), steps through the first 10 iterations of each algorithm, and plots the value "
            "function, the policy and the convergence."
        )
        left, right = st.columns([1, 3], gap="medium")
        with left:
            parameter_widgets()
            try:
                params = params_from_state()
            except ValueError as e:
                st.error(f"Fix the parameters: {e}")
                params = None
        with right:
            if params is None:
                st.info("The explorer appears once the parameters are valid.")
                return
            tol = float(st.session_state[_key("tol")])
            try:
                data = _explore_cached(json.dumps(asdict(params), sort_keys=True), tol)
            except ValueError as e:
                st.error(f"Could not solve this problem: {e}")
                return
            html = render_html(data, theme=_theme())
            components.html(html, height=HEIGHT, scrolling=True)
            st.download_button("Download this page as standalone HTML", html, file_name="shuttle_explorer.html", mime="text/html",
                               key="xp_download")
            st.caption("The same page from the command line: `python run.py dp --explorer results/dp/explore.html` "
                       "(add `--set capacity=10 …` for other parameters).")
