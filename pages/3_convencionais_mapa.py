"""
Widget: mapa das estacoes fluviometricas convencionais da ANA.

Replicado a partir do app principal (pagina "3. Monitoramento convencional"),
adaptado para embed no StoryMap — sem sidebar, sem header.

Fonte: catalogo ANAF (Carvalho & Braga, 2020) — snapshot 03/2020.
URL: /convencionais_mapa
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Convencionais ANA",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------
c1, c2 = st.columns([1, 3])
with c1:
    recorte = st.radio(
        "Operadora",
        ["Todas", "Apenas ANA"],
        index=0,
        horizontal=False,
    )
only_ana = "ANA" in recorte

with c2:
    st.caption(
        "Rede fluviométrica convencional registrada no catálogo ANAF "
        "(Carvalho & Braga, 2020). Operação convencional fornece registro "
        "diário. ★ em amarelo destaca Manuel Duarte (58585000)."
    )


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Carregando estacoes...")
def _load(only_ana: bool) -> pd.DataFrame:
    fname = "estacoes_convencionais_ana.csv" if only_ana else "estacoes_convencionais_todas.csv"
    return pd.read_csv(DATA_DIR / fname)


df = _load(only_ana)
df = df.dropna(subset=["Latitude", "Longitude"]).copy()


def _hex_to_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# Coloracao binaria: ANA vs demais operadoras
COLOR_ANA = _hex_to_rgb("#2563eb")
COLOR_OUTROS = _hex_to_rgb("#737373")
df["color"] = df["Responsib"].apply(
    lambda r: COLOR_ANA if r == "ANA" else COLOR_OUTROS
)

is_md = df["Code"].astype(str) == "58585000"
df_base = df[~is_md].copy()
df_md = df[is_md].copy()

df_base["radius"] = 4000
df_base["_da"] = df_base["DrainArea"].fillna(0).astype(float)
if len(df_md):
    df_md["_da"] = df_md["DrainArea"].fillna(0).astype(float)

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=df_base,
        get_position=["Longitude", "Latitude"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=10,
        pickable=True,
        opacity=0.85,
        stroked=True,
        filled=True,
        line_width_min_pixels=0.5,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=df_md,
        get_position=["Longitude", "Latitude"],
        get_fill_color=[0, 0, 0, 0],
        get_line_color=[234, 179, 8, 220],
        get_radius=70000,
        radius_min_pixels=18,
        radius_max_pixels=28,
        stroked=True,
        filled=False,
        line_width_min_pixels=3,
        pickable=False,
    ),
    pdk.Layer(
        "ScatterplotLayer",
        data=df_md,
        get_position=["Longitude", "Latitude"],
        get_fill_color=[234, 179, 8, 255],
        get_line_color=[0, 0, 0, 255],
        get_radius=42000,
        radius_min_pixels=10,
        radius_max_pixels=16,
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
        pickable=True,
    ),
]

view = pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.5, pitch=0)

deck = pdk.Deck(
    layers=layers,
    initial_view_state=view,
    tooltip={
        "html": (
            "<b>{Code} — {Name}</b><br/>"
            "{City}, {State}<br/>"
            "Operador: {Responsib}<br/>"
            "Anos de dados: {NYD} | Falhas: {MD}%"
        ),
        "style": {"fontSize": "12px"},
    },
    map_style="light",
)

mapa_col, leg_col = st.columns([5, 1])
with mapa_col:
    st.pydeck_chart(deck, use_container_width=True, height=520)

with leg_col:
    n_ana = int((df["Responsib"] == "ANA").sum())
    n_outros = int((df["Responsib"] != "ANA").sum())
    legend_html = (
        '<div style="padding-top:20px; font-size:12px; line-height:1.8;">'
        '<b>Operadora</b><br>'
        f'<span style="color:#2563eb;">●</span> ANA '
        f'<span style="color:#94a3b8;">({n_ana:,})</span><br>'
        f'<span style="color:#737373;">●</span> Outras '
        f'<span style="color:#94a3b8;">({n_outros:,})</span><br>'
    ).replace(",", ".")
    if len(df_md):
        legend_html += (
            '<br><b>Destacada</b><br>'
            '<span style="color:#eab308; font-size:1.2em;">◉</span> '
            'Manuel Duarte<br>'
            '<small>(58585000 — bacia de estudo)</small>'
        )
    legend_html += f'<hr style="margin:8px 0;"><small>Total: {len(df):,}</small>'.replace(",", ".")
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)
