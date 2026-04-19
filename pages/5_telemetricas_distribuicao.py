"""
Widget: distribuicao da area de drenagem das estacoes telemetricas da ANA.

Complementar ao mapa de telemetricas (pagina 1). Permite ver a distribuicao
do tamanho das bacias drenadas por estacoes telemetricas — util para
contextualizar onde se encaixa a bacia do estudo (Manuel Duarte, 3.117 km²).

Fonte: HidroInventario/ANA (abril/2026).
URL: /telemetricas_distribuicao
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Área de drenagem — telemétricas",
    page_icon="📈",
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
        "Recorte",
        ["Todas as redes", "Apenas RHN (ANA)"],
        index=1,
        horizontal=False,
    )
only_ana = "RHN" in recorte

with c2:
    st.caption(
        "Distribuição da área de drenagem das estações fluviométricas "
        "telemétricas. Eixo X em escala logarítmica para acomodar a ampla "
        "variação entre bacias pequenas e grandes bacias."
    )


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _load(only_ana: bool) -> pd.DataFrame:
    fname = "estacoes_telemetricas_ana.csv" if only_ana else "estacoes_telemetricas_todas.csv"
    return pd.read_csv(DATA_DIR / fname)


df = _load(only_ana)
da = df["DrainArea"].dropna().astype(float)
da = da[da > 0]  # log scale requer positivos

# ---------------------------------------------------------------------------
# Histograma em escala log
# ---------------------------------------------------------------------------
import numpy as np

# Bins log-espaçados entre 1 e 1e6 km²
bins = np.logspace(0, 6.5, 50)

fig = go.Figure(go.Histogram(
    x=da,
    xbins=dict(start=0, end=1e7, size=0),
    nbinsx=50,
    marker_color="#2563eb",
    opacity=0.85,
    name="Estações",
    hovertemplate="Área: %{x:,.0f} km²<br>Estações: %{y}<extra></extra>",
))

# Linha de referência: Manuel Duarte (3.117 km²)
fig.add_vline(
    x=3117, line_dash="dash", line_color="#eab308",
    annotation_text="Manuel Duarte (3.117 km²)",
    annotation_position="top right",
    annotation_font=dict(size=11, color="#854d0e"),
)

# Linhas de referência de escala
fig.add_vline(
    x=100, line_dash="dot", line_color="#cbd5e1",
    annotation_text="100 km²",
    annotation_position="top",
    annotation_font=dict(size=9, color="#64748b"),
)
fig.add_vline(
    x=10000, line_dash="dot", line_color="#cbd5e1",
    annotation_text="10.000 km²",
    annotation_position="top",
    annotation_font=dict(size=9, color="#64748b"),
)

fig.update_xaxes(
    type="log",
    title="<b>Área de drenagem (km²) — escala logarítmica</b>",
)
fig.update_layout(
    template="plotly_white",
    yaxis_title="<b>Número de estações</b>",
    height=460,
    margin=dict(l=40, r=30, t=30, b=40),
    bargap=0.05,
)

st.plotly_chart(fig, use_container_width=True)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Estações com área", f"{len(da):,}".replace(",", "."))
k2.metric("Mediana (km²)", f"{da.median():,.0f}".replace(",", "."))
k3.metric("Menores que 1.000 km²", f"{int((da < 1000).sum()):,}".replace(",", "."))
k4.metric("Maiores que 10.000 km²", f"{int((da >= 10000).sum()):,}".replace(",", "."))

st.caption(
    f"**Fonte:** HidroInventário/ANA (consulta em abril/2026) — recorte "
    f"**{'Apenas RHN' if only_ana else 'Todas as redes'}**. "
    "Estações sem área de drenagem registrada foram excluídas do histograma."
)
