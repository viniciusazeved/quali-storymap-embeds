"""
Widget: mapa das estacoes fluviometricas telemetricas da ANA.

Consumido pelo StoryMap ArcGIS na secao "A rede telemetrica".
URL: /mapa_telemetricas
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Telemetricas ANA",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Controles minimos — no topo, em linha unica
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    recorte = st.radio(
        "Recorte",
        ["Todas as redes", "Apenas RHN (ANA)"],
        index=1,
        horizontal=False,
        label_visibility="visible",
    )
only_ana = "RHN" in recorte

with c2:
    da_min = st.number_input(
        "Área mínima (km²)", min_value=0, max_value=5000, value=0, step=500,
    )

with c3:
    st.caption(
        "Rede fluviométrica telemétrica do Brasil — transmissão automática, "
        "vazão horária ou sub-horária. Fonte: HidroWeb/ANA (abril/2026). "
        "★ amarelo destaca a bacia do estudo (Manuel Duarte)."
    )

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Carregando estacoes...")
def _load(only_ana: bool) -> pd.DataFrame:
    fname = "estacoes_telemetricas_ana.csv" if only_ana else "estacoes_telemetricas_todas.csv"
    return pd.read_csv(DATA_DIR / fname)


df = _load(only_ana)
df = df.dropna(subset=["Latitude", "Longitude"]).copy()
if da_min > 0:
    df = df[df["DrainArea"].fillna(0) >= da_min]
df["_da"] = df["DrainArea"].fillna(0).astype(float)


# ---------------------------------------------------------------------------
# Estilo — cores e tamanhos
# ---------------------------------------------------------------------------
def _hex_to_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


if only_ana:
    COLORS = {"Ativo": "#16a34a", "Manutenção": "#f59e0b"}
    df["_cat"] = df["Status"].fillna("(sem status)")
else:
    COLORS = {
        "RHN": "#2563eb",
        "Setor Elétrico": "#dc2626",
        "Setores Regulados": "#9333ea",
        "Açudes Semiárido": "#f59e0b",
        "CotaOnline": "#0ea5e9",
        "Inpe/Sivam(desativadas)": "#737373",
    }
    df["_cat"] = df["Origem"].fillna("(sem origem)")

df["color"] = df["_cat"].map(lambda c: _hex_to_rgb(COLORS.get(c, "#737373")))

# Separar Manuel Duarte para layers proprios (z-order)
is_md = df["Code"] == "58585000"
df_base = df[~is_md].copy()
df_md = df[is_md].copy()

df_base["radius"] = 4000

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
            "Rio: {River}<br/>"
            "Área: {_da} km²<br/>"
            "Origem: {Origem} · Status: {Status}"
        ),
        "style": {"fontSize": "12px"},
    },
    map_style="light",
)

# ---------------------------------------------------------------------------
# Mapa + legenda compacta
# ---------------------------------------------------------------------------
mapa_col, leg_col = st.columns([5, 1])
with mapa_col:
    st.pydeck_chart(deck, use_container_width=True, height=520)

with leg_col:
    legend_html = '<div style="padding-top:20px; font-size:12px; line-height:1.8;">'
    legend_html += '<b>Categoria</b><br>'
    for cat, color in COLORS.items():
        if cat in df["_cat"].unique():
            n = int((df["_cat"] == cat).sum())
            legend_html += (
                f'<span style="color:{color};">●</span> {cat} '
                f'<span style="color:#94a3b8;">({n:,})</span><br>'.replace(",", ".")
            )
    n_md = int(is_md.sum())
    if n_md > 0:
        legend_html += (
            '<br><b>Destacada</b><br>'
            '<span style="color:#eab308; font-size:1.2em;">◉</span> '
            'Manuel Duarte<br>'
            '<small>(58585000 — bacia do estudo)</small>'
        )
    legend_html += f'<hr style="margin:8px 0;"><small>Total: {len(df):,}</small>'.replace(",", ".")
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)
