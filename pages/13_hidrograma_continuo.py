"""
Widget: hidrograma observado x predito em serie temporal continua.

Widget enxuto focado na visualizacao da serie horaria completa. Horizonte
fixo em 1 h (mais proximo da operacao continua). Substitui parte do
explorador_hidrogramas, reduzindo peso para carregamento rapido no StoryMap.

URL: /hidrograma_continuo
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
    page_title="Hidrograma contínuo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_FILE = Path(__file__).parent.parent / "data" / "hidrogramas.csv"

MODEL_OPTIONS = {
    "LSTM + TTD Base (aprendível)": "Base",
    "LSTM + TTD Base Fixed (parâmetros fixos)": "Base_Fixed",
    "LSTM + TTD Manning (aprendível)": "Manning",
}


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    return df.set_index("timestamp").sort_index()


df = _load()
ts_min = df.index.min().to_pydatetime()
ts_max = df.index.max().to_pydatetime()

# Controles enxutos
c1, c2 = st.columns([3, 1])
with c1:
    modelo_label = st.selectbox(
        "Configuração",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )
    modelo_key = MODEL_OPTIONS[modelo_label]
with c2:
    show_precip = st.checkbox("Precipitação", value=True)

default_start = max(pd.Timestamp("2025-09-01").to_pydatetime(), ts_min)
janela = st.slider(
    "Janela temporal",
    min_value=ts_min,
    max_value=ts_max,
    value=(default_start, ts_max),
    format="YYYY-MM-DD",
)

mask = (df.index >= janela[0]) & (df.index <= janela[1])
dfw = df.loc[mask]

obs = dfw["Q_obs_1h"]
pred = dfw[f"Q_pred_{modelo_key}_1h"]
precip = dfw["P_mean"]

# KPIs
valid = (~obs.isna()) & (~pred.isna())
o = obs[valid].values
p = pred[valid].values

if len(o) > 10:
    nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    pbias = float(100 * (p.sum() - o.sum()) / o.sum())
    r2 = float(np.corrcoef(o, p)[0, 1]) ** 2
else:
    nse = rmse = pbias = r2 = float("nan")

k1, k2, k3, k4 = st.columns(4)
k1.metric("NSE", f"{nse:.3f}" if np.isfinite(nse) else "—")
k2.metric("RMSE (m³/s)", f"{rmse:.2f}" if np.isfinite(rmse) else "—")
k3.metric("PBIAS (%)", f"{pbias:+.2f}" if np.isfinite(pbias) else "—")
k4.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "—")

st.divider()

# Hidrograma
if show_precip:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.22, 0.78], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Bar(
            x=precip.index, y=precip.values,
            marker_color="#0ea5e9", opacity=0.7,
            name="Precipitação",
            hovertemplate="%{x|%d/%m %H:%M}<br>P = %{y:.2f} mm/h<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title_text="P (mm/h)", autorange="reversed",
                     row=1, col=1, gridcolor="#f1f5f9")
    row_q = 2
else:
    fig = go.Figure()
    row_q = None

t_obs = go.Scatter(
    x=obs.index, y=obs.values,
    mode="lines", line=dict(color="#0f172a", width=1.8),
    name="Observado",
    hovertemplate="%{x|%d/%m %H:%M}<br>Q_obs = %{y:.1f} m³/s<extra></extra>",
)
t_pred = go.Scatter(
    x=pred.index, y=pred.values,
    mode="lines", line=dict(color="#2563eb", width=1.8),
    name="Predito",
    hovertemplate="%{x|%d/%m %H:%M}<br>Q_pred = %{y:.1f} m³/s<extra></extra>",
)

if row_q is not None:
    fig.add_trace(t_obs, row=row_q, col=1)
    fig.add_trace(t_pred, row=row_q, col=1)
    fig.update_yaxes(title_text="Q (m³/s)", row=row_q, col=1, gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", row=row_q, col=1, gridcolor="#f1f5f9")
else:
    fig.add_trace(t_obs)
    fig.add_trace(t_pred)
    fig.update_yaxes(title_text="Q (m³/s)", gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", gridcolor="#f1f5f9")

fig.update_layout(
    template="plotly_white",
    height=460,
    hovermode="x unified",
    margin=dict(l=40, r=20, t=20, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)",
    ),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"**Configuração:** {modelo_label} · **Horizonte:** 1 h à frente · "
    f"**Amostras válidas:** {len(o):,} h."
    .replace(",", ".")
)
