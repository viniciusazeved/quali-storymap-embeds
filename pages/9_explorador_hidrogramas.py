"""
Widget: explorador de hidrogramas observado x predito por modelo e evento.

Permite ao leitor navegar pelo periodo de teste (abr/2025 a dez/2025), escolher
uma das tres configuracoes campeas (Base, Base_Fixed, Manning), selecionar um
horizonte de previsao (1h, 6h ou 24h) e recortar uma janela temporal. O grafico
mostra Q_obs x Q_pred, com precipitacao media areal opcional no topo; KPIs
(NSE, RMSE, pico observado, pico predito) sao recalculados para a janela.

Consumido pelo StoryMap ArcGIS na secao "Previsao de vazoes nos eventos".
URL: /explorador_hidrogramas
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Explorador de hidrogramas",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_FILE = Path(__file__).parent.parent / "data" / "hidrogramas.csv"

MODEL_OPTIONS = {
    "LSTM + TTD Base (aprendível)": "Base",
    "LSTM + TTD Base (fixo)": "Base_Fixed",
    "LSTM + TTD Manning (aprendível)": "Manning",
}
HORIZON_OPTIONS = ["1h", "6h", "24h"]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


df = _load()
ts_min = df.index.min().to_pydatetime()
ts_max = df.index.max().to_pydatetime()


# ---------------------------------------------------------------------------
# Controles no topo
# ---------------------------------------------------------------------------
st.markdown(
    "##### Observado × predito no período de teste — abr/2025 a dez/2025"
)

c1, c2, c3 = st.columns([2, 1.5, 1])
with c1:
    modelo_label = st.selectbox(
        "Configuração",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )
    modelo_key = MODEL_OPTIONS[modelo_label]
with c2:
    horizonte = st.radio(
        "Horizonte de previsão",
        HORIZON_OPTIONS,
        index=1,
        horizontal=True,
    )
with c3:
    show_precip = st.checkbox("Precipitação", value=True)

# Slider de janela temporal
default_start = pd.Timestamp("2025-09-01").to_pydatetime()
default_start = max(default_start, ts_min)
janela = st.slider(
    "Janela temporal",
    min_value=ts_min,
    max_value=ts_max,
    value=(default_start, ts_max),
    format="YYYY-MM-DD",
)


# ---------------------------------------------------------------------------
# Filtra janela
# ---------------------------------------------------------------------------
mask = (df.index >= janela[0]) & (df.index <= janela[1])
dfw = df.loc[mask].copy()

obs_col = f"Q_obs_{horizonte}"
pred_col = f"Q_pred_{modelo_key}_{horizonte}"
obs = dfw[obs_col]
pred = dfw[pred_col]
precip = dfw["P_mean"] if show_precip else None


# ---------------------------------------------------------------------------
# Grafico
# ---------------------------------------------------------------------------
if show_precip:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.22, 0.78],
        vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Bar(
            x=precip.index, y=precip.values,
            marker_color="#0ea5e9", opacity=0.65,
            name="Precipitação",
            hovertemplate="%{x|%d/%m/%Y %H:%M}<br>P = %{y:.2f} mm/h<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="<b>P (mm/h)</b>", autorange="reversed",
        row=1, col=1, gridcolor="#f1f5f9",
    )
    row_q = 2
else:
    fig = go.Figure()
    row_q = None

trace_obs = go.Scatter(
    x=obs.index, y=obs.values,
    mode="lines", line=dict(color="#0f172a", width=2),
    name="Observado",
    hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Q_obs = %{y:.1f} m³/s<extra></extra>",
)
trace_pred = go.Scatter(
    x=pred.index, y=pred.values,
    mode="lines", line=dict(color="#2563eb", width=2),
    name=f"Predito — {modelo_label.split(' — ')[0]}",
    hovertemplate="%{x|%d/%m/%Y %H:%M}<br>Q_pred = %{y:.1f} m³/s<extra></extra>",
)

if row_q is not None:
    fig.add_trace(trace_obs, row=row_q, col=1)
    fig.add_trace(trace_pred, row=row_q, col=1)
    fig.update_yaxes(title_text="<b>Q (m³/s)</b>", row=row_q, col=1, gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", row=row_q, col=1, gridcolor="#f1f5f9")
else:
    fig.add_trace(trace_obs)
    fig.add_trace(trace_pred)
    fig.update_yaxes(title_text="<b>Q (m³/s)</b>", gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", gridcolor="#f1f5f9")

fig.update_layout(
    template="plotly_white",
    height=500,
    hovermode="x unified",
    margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)",
    ),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# KPIs do periodo
# ---------------------------------------------------------------------------
valid = (~obs.isna()) & (~pred.isna())
o = obs[valid].values
p = pred[valid].values

if len(o) > 10:
    nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    pico_obs = float(o.max())
    pico_pred = float(p.max())
    err_pico = 100.0 * (pico_pred - pico_obs) / pico_obs if pico_obs > 0 else float("nan")
else:
    nse = rmse = pico_obs = pico_pred = err_pico = float("nan")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("NSE (janela)", f"{nse:.3f}" if np.isfinite(nse) else "—")
k2.metric("RMSE (m³/s)", f"{rmse:.2f}" if np.isfinite(rmse) else "—")
k3.metric("Pico observado (m³/s)", f"{pico_obs:.1f}" if np.isfinite(pico_obs) else "—")
k4.metric("Pico predito (m³/s)", f"{pico_pred:.1f}" if np.isfinite(pico_pred) else "—")
k5.metric("Erro do pico (%)", f"{err_pico:+.1f}" if np.isfinite(err_pico) else "—")


st.caption(
    f"**Configuração:** {modelo_label} · **Horizonte:** {horizonte} à frente · "
    f"**Amostras na janela:** {len(o):,}".replace(",", ".")
    + ". Ajuste o intervalo para avaliar eventos específicos de cheia ou "
    "períodos de recessão. O NSE exibido corresponde apenas à janela recortada; "
    "valores globais por configuração estão no widget de trade-off."
)
