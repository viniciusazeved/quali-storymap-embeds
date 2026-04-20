"""
Widget: mapa das estacoes fluviometricas convencionais da ANA — versao enxuta.

Mostra apenas estacoes RHN/ANA por padrao, com plotly scattermapbox (mais
leve que pydeck) para reduzir cold start no Streamlit Cloud. Manuel Duarte
destacada em amarelo.

Fonte: catalogo ANAF (Carvalho & Braga, 2020).
URL: /convencionais_mapa
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
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


@st.cache_data(ttl=3600, show_spinner="Carregando estações...")
def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "estacoes_convencionais_ana.csv")
    return df.dropna(subset=["Latitude", "Longitude"])


df = _load()

st.caption(
    "Rede fluviométrica convencional registrada no catálogo ANAF "
    "(Carvalho & Braga, 2020) — apenas estações sob responsabilidade da ANA. "
    "Operação convencional fornece registro diário. ★ em amarelo destaca "
    "Manuel Duarte (58585000)."
)

# Separar Manuel Duarte
is_md = df["Code"].astype(str) == "58585000"
df_base = df[~is_md]
df_md = df[is_md]

# Hover simplificado
hover_base = (
    df_base["Name"].fillna("—").astype(str) + "<br>"
    + df_base["City"].fillna("—").astype(str) + " / "
    + df_base["State"].fillna("—").astype(str) + "<br>"
    + "Código: " + df_base["Code"].astype(str)
)

fig = go.Figure()
fig.add_trace(go.Scattermapbox(
    lat=df_base["Latitude"],
    lon=df_base["Longitude"],
    mode="markers",
    marker=dict(size=5, color="#2563eb", opacity=0.7),
    text=hover_base,
    hoverinfo="text",
    name=f"ANA ({len(df_base):,})".replace(",", "."),
))

if len(df_md):
    fig.add_trace(go.Scattermapbox(
        lat=df_md["Latitude"],
        lon=df_md["Longitude"],
        mode="markers",
        marker=dict(size=16, color="#eab308", opacity=1.0),
        text=["★ Manuel Duarte (58585000) — bacia de estudo"] * len(df_md),
        hoverinfo="text",
        name="Manuel Duarte",
    ))

fig.update_layout(
    mapbox=dict(
        style="carto-positron",
        center=dict(lat=-14.2, lon=-51.9),
        zoom=3.2,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=540,
    legend=dict(
        yanchor="top", y=0.99, xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.85)",
    ),
    showlegend=True,
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"**Total:** {len(df):,} estações ANA. Fonte: ANAF / ANA."
    .replace(",", ".")
)
