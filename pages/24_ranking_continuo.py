"""
Widget: ranking das 10 configuracoes por NSE na simulacao continua.

Contraparte do /ranking_nse (que e por horizonte de previsao) mas com
o recorte da simulacao continua de 9 meses. Barras horizontais
ordenadas, escala de cor igual ao ranking de previsao.

Destaque: o ranking da continua reordena configuracoes face ao
ranking da previsao 6 h. Modelos com parametros fixos (antes em
posicoes 5-8 na previsao) sobem para o topo; modelos com parametros
aprendiveis descem.

URL: /ranking_continuo
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Ranking NSE contínua — TTD-SCS-LSTM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    with open(DATA_DIR / "summary_continuous.json", encoding="utf-8") as f:
        raw = json.load(f)
    rows = [
        {
            "Modelo": m["model_name"],
            "NSE": m["metrics"]["nse"],
            "KGE": m["metrics"]["kge"],
            "RMSE": m["metrics"]["rmse"],
            "PBIAS": m["metrics"]["pbias"],
        }
        for m in raw
    ]
    return pd.DataFrame(rows)


df = _load()
d = df.sort_values("NSE", ascending=True).copy()

# ---------------------------------------------------------------------------
# Grafico de barras horizontais (estilo do app principal)
# ---------------------------------------------------------------------------
fig = go.Figure(go.Bar(
    x=d["NSE"],
    y=d["Modelo"],
    orientation="h",
    marker=dict(
        color=d["NSE"],
        colorscale=[[0, "#fee2e2"], [0.5, "#fef3c7"], [1, "#15803d"]],
        showscale=False,
    ),
    text=d["NSE"].round(3),
    textposition="outside",
    customdata=d[["KGE", "RMSE", "PBIAS"]].values,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "NSE: %{x:.3f}<br>"
        "KGE: %{customdata[0]:.3f}<br>"
        "RMSE: %{customdata[1]:.2f} m³/s<br>"
        "PBIAS: %{customdata[2]:+.2f}%"
        "<extra></extra>"
    ),
))

fig.update_layout(
    title="Ranking de modelos — NSE em simulação contínua",
    template="plotly_white",
    xaxis_title="NSE",
    xaxis=dict(range=[0.4, 0.9]),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Ranking dos 10 modelos pelo NSE da simulação contínua de nove meses "
    "(mar–dez 2025). Três padrões se destacam: (1) o `LSTM_TTD_Base_Fixed` "
    "lidera com NSE = 0,824, à frente do `LSTM_TTD_Manning` (NSE = 0,809) — "
    "inversão em relação ao ranking de previsão 6 h, onde o Base (ajustável) "
    "lidera; (2) configurações com parâmetros **fixos** ocupam metade do pódio, "
    "confirmando o efeito regularizador da física codificada; (3) o "
    "`LSTM_TTD_Manning_Fixed` desce ao último lugar (NSE = 0,541), abaixo "
    "até do `LSTM_Lumped` (NSE = 0,607), mostrando que física mal calibrada "
    "pode ser pior que física ausente."
)
