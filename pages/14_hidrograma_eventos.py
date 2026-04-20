"""
Widget: catalogo de eventos de cheia — forecasting 6 h.

Detecta automaticamente picos acima do Q95 com separacao de 48 h,
permite selecionar um evento e exibe zoom (72 h antes, 120 h depois).
Substitui a aba de eventos do explorador_hidrogramas em widget autonomo
e enxuto para carregamento rapido no StoryMap.

URL: /hidrograma_eventos
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
    page_title="Eventos de cheia",
    page_icon="🌊",
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

obs = df["Q_obs_6h"]
pred = df[f"Q_pred_{modelo_key}_6h"]
precip = df["P_mean"]

# Q95 e detecção de picos
q95 = float(np.nanpercentile(obs.dropna().values, 95))
obs_arr = obs.values.copy()
obs_arr_filled = np.where(np.isnan(obs_arr), -np.inf, obs_arr)

peaks = []
i = 1
while i < len(obs_arr_filled) - 1:
    if (obs_arr_filled[i] > q95
            and obs_arr_filled[i] > obs_arr_filled[i - 1]
            and obs_arr_filled[i] > obs_arr_filled[i + 1]):
        peaks.append(i)
        i += 48
    else:
        i += 1

st.markdown(
    f"**Q95 = {q95:.1f} m³/s** · **{len(peaks)} evento(s)** detectado(s) "
    f"com pico acima do Q95 (separação mínima de 48 h)."
)

if not peaks:
    st.info("Nenhum evento de cheia detectado. Verifique os dados de entrada.")
    st.stop()

event_options = [
    f"Evento {k + 1}: {obs.index[p]:%d/%m/%Y %H:%M}  ·  Q_pico = {obs_arr[p]:.1f} m³/s"
    for k, p in enumerate(peaks)
]
evt = st.selectbox("Selecione um evento", event_options, index=0)
idx = event_options.index(evt)
peak_pos = peaks[idx]

# Zoom 72h antes, 120h depois
i0 = max(0, peak_pos - 72)
i1 = min(len(obs), peak_pos + 120)
obs_e = obs.iloc[i0:i1]
pred_e = pred.iloc[i0:i1]
precip_e = precip.iloc[i0:i1]

# KPIs do evento
o_valid = obs_e.dropna()
o_e = o_valid.values
p_e = pred_e.reindex(o_valid.index).values
valid_e = ~np.isnan(p_e)
o_e = o_e[valid_e]
p_e = p_e[valid_e]

if len(o_e) > 10:
    nse_e = 1 - np.sum((o_e - p_e) ** 2) / np.sum((o_e - o_e.mean()) ** 2)
    pbias_e = 100 * (p_e.sum() - o_e.sum()) / o_e.sum()
    pico_o = float(o_e.max())
    pico_p = float(p_e.max())
    err_p = 100 * (pico_p - pico_o) / pico_o if pico_o > 0 else float("nan")
    vol_o = float(np.trapezoid(o_e))
    vol_p = float(np.trapezoid(p_e))
    err_vol = 100 * (vol_p - vol_o) / vol_o if vol_o > 0 else float("nan")
else:
    nse_e = pbias_e = pico_o = pico_p = err_p = err_vol = float("nan")

ce1, ce2, ce3, ce4, ce5 = st.columns(5)
ce1.metric("NSE", f"{nse_e:.3f}" if np.isfinite(nse_e) else "—")
ce2.metric("PBIAS (%)", f"{pbias_e:+.2f}" if np.isfinite(pbias_e) else "—")
ce3.metric("Pico obs (m³/s)", f"{pico_o:.1f}" if np.isfinite(pico_o) else "—")
ce4.metric("Pico pred (m³/s)", f"{pico_p:.1f}" if np.isfinite(pico_p) else "—")
ce5.metric("Erro pico (%)", f"{err_p:+.1f}" if np.isfinite(err_p) else "—")

st.divider()

# Hidrograma do evento
if show_precip:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.22, 0.78], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Bar(
            x=precip_e.index, y=precip_e.values,
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
    x=obs_e.index, y=obs_e.values,
    mode="lines", line=dict(color="#0f172a", width=2.2),
    name="Observado",
    hovertemplate="%{x|%d/%m %H:%M}<br>Q_obs = %{y:.1f} m³/s<extra></extra>",
)
t_pred = go.Scatter(
    x=pred_e.index, y=pred_e.values,
    mode="lines", line=dict(color="#2563eb", width=2.2),
    name="Predito (6 h)",
    hovertemplate="%{x|%d/%m %H:%M}<br>Q_pred = %{y:.1f} m³/s<extra></extra>",
)

# Linha vertical no pico
peak_time = obs.index[peak_pos]

if row_q is not None:
    fig.add_trace(t_obs, row=row_q, col=1)
    fig.add_trace(t_pred, row=row_q, col=1)
    fig.add_vline(x=peak_time, line_dash="dot", line_color="#dc2626",
                  opacity=0.6, row=row_q, col=1)
    fig.update_yaxes(title_text="Q (m³/s)", row=row_q, col=1, gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", row=row_q, col=1, gridcolor="#f1f5f9")
else:
    fig.add_trace(t_obs)
    fig.add_trace(t_pred)
    fig.add_vline(x=peak_time, line_dash="dot", line_color="#dc2626", opacity=0.6)
    fig.update_yaxes(title_text="Q (m³/s)", gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", gridcolor="#f1f5f9")

fig.update_layout(
    template="plotly_white",
    title=f"Evento em {peak_time:%d de %B de %Y, %H:%M}",
    height=440,
    hovermode="x unified",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)",
    ),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Eventos isolados a partir de máximos locais acima do Q95, com separação "
    "mínima de 48 h. Visualização de 72 h antes e 120 h depois do pico observado. "
    f"Configuração: {modelo_label} · Horizonte: 6 h. "
    "Erro do pico positivo = superestimativa da predição."
)
