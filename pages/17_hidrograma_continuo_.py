"""
Widget: hidrograma continuo comparando melhor, baseline simples e pior.

Variante do /hidrograma_continuo focada em contraste entre as tres
configuracoes extremas da simulacao continua (9 meses):
- Melhor: LSTM_TTD_Base_Fixed (NSE = 0.824, fisica calibrada)
- Baseline simples: LSTM_Lumped (NSE = 0.607, concentrado sem fisica)
- Pior: LSTM_TTD_Manning_Fixed (NSE = 0.541, fisica mal calibrada)

Visualiza simultaneamente as tres series preditas + serie observada,
com KPIs por modelo. Permite janela temporal ajustavel.

URL: /hidrograma_continuo_
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
    page_title="Hidrograma contínuo — melhor × pior",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_FILE = Path(__file__).parent.parent / "data" / "continuo_3modelos.json"

# Configuracoes em ordem de desempenho (melhor -> pior)
MODELOS = [
    {
        "coluna": "Q_sim_Base_Fixed",
        "label": "Melhor — Base Fixed",
        "descricao": "LSTM + TTD Base, parâmetros fixos",
        "cor": "#16a34a",  # verde
    },
    {
        "coluna": "Q_sim_Lumped",
        "label": "Baseline — Lumped",
        "descricao": "LSTM concentrado, sem física",
        "cor": "#94a3b8",  # cinza ardosia
    },
    {
        "coluna": "Q_sim_Manning_Fixed",
        "label": "Pior — Manning Fixed",
        "descricao": "LSTM + TTD Manning, Tc em escala 1",
        "cor": "#dc2626",  # vermelho
    },
]


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    import json
    with open(DATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    df = pd.DataFrame({
        "Q_obs": raw["Q_obs"],
        "Q_sim_Base_Fixed": raw["Q_sim_Base_Fixed"],
        "Q_sim_Lumped": raw["Q_sim_Lumped"],
        "Q_sim_Manning_Fixed": raw["Q_sim_Manning_Fixed"],
        "P_mean": raw["P_mean"],
    }, index=pd.to_datetime(raw["timestamps"]))
    return df.sort_index()


df = _load()
ts_min = df.index.min().to_pydatetime()
ts_max = df.index.max().to_pydatetime()

# Controles enxutos
show_precip = st.checkbox("Mostrar precipitação", value=True)

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
obs = dfw["Q_obs"]
precip = dfw["P_mean"]


def _metricas(o: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(o) < 10:
        return {k: float("nan") for k in ("nse", "kge", "rmse", "pbias", "r2")}
    nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    pbias = float(100 * (p.sum() - o.sum()) / o.sum())
    r2 = float(np.corrcoef(o, p)[0, 1]) ** 2
    # KGE (Gupta et al., 2009) — usa correlacao de Pearson, razao de variancias e vies
    r = float(np.corrcoef(o, p)[0, 1])
    alpha = float(np.std(p) / np.std(o)) if np.std(o) > 0 else float("nan")
    beta = float(np.mean(p) / np.mean(o)) if np.mean(o) > 0 else float("nan")
    kge = 1 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"nse": float(nse), "kge": kge, "rmse": rmse, "pbias": pbias, "r2": r2}


# KPIs por modelo
kpi_cols = st.columns(len(MODELOS))
for col, cfg in zip(kpi_cols, MODELOS):
    pred = dfw[cfg["coluna"]]
    valid = (~obs.isna()) & (~pred.isna())
    m = _metricas(obs[valid].values, pred[valid].values)
    with col:
        st.markdown(
            f"<div style='border-left:4px solid {cfg['cor']};padding:6px 10px;"
            f"background:#f8fafc;border-radius:4px;'>"
            f"<div style='font-weight:600;color:#0f172a;font-size:13px;'>{cfg['label']}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-bottom:4px;'>{cfg['descricao']}</div>"
            f"<div style='font-size:12px;line-height:1.5;'>"
            f"NSE: <b>{m['nse']:.3f}</b> &nbsp;·&nbsp; "
            f"KGE: <b>{m['kge']:.3f}</b><br>"
            f"RMSE: {m['rmse']:.1f} m³/s &nbsp;·&nbsp; "
            f"PBIAS: {m['pbias']:+.1f}%"
            f"</div></div>",
            unsafe_allow_html=True,
        )

st.divider()

# Hidrograma sobreposto
if show_precip:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.20, 0.80], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Bar(
            x=precip.index, y=precip.values,
            marker_color="#0ea5e9", opacity=0.7,
            name="Precipitação",
            hovertemplate="%{x|%d/%m %H:%M}<br>P = %{y:.2f} mm/h<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title_text="P (mm/h)", autorange="reversed",
                     row=1, col=1, gridcolor="#f1f5f9")
    row_q = 2
else:
    fig = go.Figure()
    row_q = None

# Observado em preto grosso
t_obs = go.Scatter(
    x=obs.index, y=obs.values,
    mode="lines", line=dict(color="#0f172a", width=2.0),
    name="Observado",
    hovertemplate="%{x|%d/%m %H:%M}<br>Q_obs = %{y:.1f} m³/s<extra></extra>",
)
if row_q is not None:
    fig.add_trace(t_obs, row=row_q, col=1)
else:
    fig.add_trace(t_obs)

# Tres simulacoes sobrepostas
for cfg in MODELOS:
    pred = dfw[cfg["coluna"]]
    tr = go.Scatter(
        x=pred.index, y=pred.values,
        mode="lines", line=dict(color=cfg["cor"], width=1.6),
        name=cfg["label"],
        opacity=0.85,
        hovertemplate=(
            f"<b>{cfg['label']}</b><br>"
            "%{x|%d/%m %H:%M}<br>"
            "Q_sim = %{y:.1f} m³/s"
            "<extra></extra>"
        ),
    )
    if row_q is not None:
        fig.add_trace(tr, row=row_q, col=1)
    else:
        fig.add_trace(tr)

if row_q is not None:
    fig.update_yaxes(title_text="Q (m³/s)", row=row_q, col=1, gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", row=row_q, col=1, gridcolor="#f1f5f9")
else:
    fig.update_yaxes(title_text="Q (m³/s)", gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", gridcolor="#f1f5f9")

fig.update_layout(
    template="plotly_white",
    height=500,
    hovermode="x unified",
    margin=dict(l=40, r=20, t=30, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)",
    ),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Simulação contínua de 9 meses (período de teste), rodada ponto-a-ponto "
    "sem reset. A configuração `Base Fixed` (melhor) mantém fidelidade ao "
    "hidrograma observado; o `Lumped` — apesar da simplicidade — preserva o "
    "padrão geral das recessões; o `Manning Fixed` apresenta derivas "
    "acentuadas em períodos secos e de transição, evidenciando que uma "
    "parametrização física inadequada pode deteriorar o desempenho abaixo "
    "do baseline mais simples."
)
