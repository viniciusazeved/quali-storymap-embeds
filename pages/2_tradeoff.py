"""
Widget: trade-off previsao 6h x simulacao continua por configuracao.

Mostra os 10 modelos do estudo comparativo (ablacao) num scatter:
- Eixo X: NSE na previsao de 6 horas (metrica de forecasting)
- Eixo Y: NSE na simulacao continua de 9 meses
- Destaque visual para os 3 modelos campeoes em cada dimensao.

Consumido pelo StoryMap ArcGIS na secao "O trade-off entre previsao e
simulacao". URL: /tradeoff
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Trade-off — TTD-SCS-LSTM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    forecast = pd.read_csv(DATA_DIR / "summary.csv")

    with open(DATA_DIR / "summary_continuous.json", encoding="utf-8") as f:
        continuous_raw = json.load(f)
    cont_rows = [
        {"Modelo": m["model_name"], "NSE_cont": m["metrics"]["nse"]}
        for m in continuous_raw
    ]
    continuous = pd.DataFrame(cont_rows)

    df = forecast.merge(continuous, on="Modelo", how="inner")
    df["label_curto"] = df["Modelo"].str.replace("LSTM_TTD_", "").str.replace("_", " ")
    df["label_curto"] = df["label_curto"].replace({"LSTM": "LSTM puro", "LSTM Lumped": "Lumped"})
    return df


df = _load()

# Configuracoes de referencia — desempenho pareto-otimo em cada modo
HIGHLIGHTS = {
    "LSTM_TTD_Base": "Maior NSE em previsão 6h",
    "LSTM_TTD_Base_Fixed": "Maior NSE em simulação contínua",
    "LSTM_TTD_Manning": "NSE ≥ 0,81 em ambos os modos",
}

# ---------------------------------------------------------------------------
# Cabecalho minimo
# ---------------------------------------------------------------------------
st.markdown(
    "##### Desempenho das 10 configurações do estudo comparativo — bacia do Rio Preto"
)

# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------
fig = go.Figure()

# Linha diagonal y = x (referencia de equivalencia entre os dois modos)
lims = [0.40, 0.90]
fig.add_trace(go.Scatter(
    x=lims, y=lims,
    mode="lines",
    line=dict(color="#cbd5e1", dash="dash", width=1.5),
    name="y = x (equivalência)",
    hoverinfo="skip",
    showlegend=False,
))

# Pontos
for _, row in df.iterrows():
    modelo = row["Modelo"]
    is_highlight = modelo in HIGHLIGHTS
    fig.add_trace(go.Scatter(
        x=[row["NSE_6h"]],
        y=[row["NSE_cont"]],
        mode="markers+text",
        text=[row["label_curto"]],
        textposition="top center",
        textfont=dict(
            size=11 if is_highlight else 10,
            color="#0f172a" if is_highlight else "#475569",
            family="sans-serif",
        ),
        marker=dict(
            size=20 if is_highlight else 12,
            color="#16a34a" if is_highlight else "#2563eb",
            line=dict(color="black", width=1.5 if is_highlight else 0.6),
            symbol="star" if is_highlight else "circle",
        ),
        name=row["label_curto"],
        customdata=[[
            row["Modelo"],
            row["NSE_6h"],
            row["NSE_cont"],
            HIGHLIGHTS.get(modelo, "—"),
        ]],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "NSE previsão 6h: %{customdata[1]:.3f}<br>"
            "NSE simulação contínua: %{customdata[2]:.3f}<br>"
            "Destaque: %{customdata[3]}"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

# Anotacoes nos quadrantes — descricao tecnica do comportamento
fig.add_annotation(
    x=0.88, y=0.43, xref="x", yref="y",
    text="<b>Alto NSE apenas em previsão</b><br><span style='color:#64748b;'>acúmulo de erros na simulação contínua</span>",
    showarrow=False, font=dict(size=10, color="#334155"),
    bgcolor="rgba(255,255,255,0.85)", borderpad=4,
    xanchor="right", yanchor="bottom",
)
fig.add_annotation(
    x=0.43, y=0.88, xref="x", yref="y",
    text="<b>Alto NSE apenas em simulação contínua</b><br><span style='color:#64748b;'>previsão de curto prazo aquém</span>",
    showarrow=False, font=dict(size=10, color="#334155"),
    bgcolor="rgba(255,255,255,0.85)", borderpad=4,
    xanchor="left", yanchor="top",
)
fig.add_annotation(
    x=0.88, y=0.88, xref="x", yref="y",
    text="<b>NSE ≥ 0,75 em ambos os modos</b><br><span style='color:#64748b;'>classificação <i>Muito Bom</i> de Moriasi (2007)</span>",
    showarrow=False, font=dict(size=10, color="#166534"),
    bgcolor="rgba(240, 253, 244, 0.9)", borderpad=4,
    xanchor="right", yanchor="top",
)

# Limites NSE=0,75 (classe "Bom" Moriasi) como linhas de corte
fig.add_shape(
    type="line", x0=0.75, x1=0.75, y0=lims[0], y1=lims[1],
    line=dict(color="#cbd5e1", dash="dot", width=1),
)
fig.add_shape(
    type="line", x0=lims[0], x1=lims[1], y0=0.75, y1=0.75,
    line=dict(color="#cbd5e1", dash="dot", width=1),
)

fig.update_layout(
    template="plotly_white",
    xaxis=dict(
        title="<b>NSE — previsão 6 horas à frente</b><br><sup>janela deslizante sobre o período de teste</sup>",
        range=lims,
        gridcolor="#f1f5f9",
    ),
    yaxis=dict(
        title="<b>NSE — simulação contínua</b><br><sup>roda ponto-a-ponto sem reset por 9 meses</sup>",
        range=lims,
        gridcolor="#f1f5f9",
    ),
    height=560,
    margin=dict(l=20, r=20, t=10, b=20),
    plot_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "**Leitura do gráfico.** A diagonal representa equivalência entre os dois "
    "modos de avaliação; pontos acima indicam NSE superior em simulação contínua, "
    "pontos abaixo indicam NSE superior em previsão multi-horizonte. As linhas "
    "pontilhadas marcam o limiar NSE = 0,75, correspondente à classe *Muito Bom* "
    "de Moriasi (2007). As três configurações destacadas correspondem aos "
    "resultados pareto-ótimos em cada dimensão: `LSTM_TTD_Base` (parâmetros "
    "aprendíveis) para previsão de curto prazo, `LSTM_TTD_Base_Fixed` (parâmetros "
    "fixos) para simulação contínua, e `LSTM_TTD_Manning` para aplicações que "
    "exigem desempenho consistente em ambos os modos."
)
