"""
Widget: resultados da Fase 1 — estudo comparativo (ablacao) das 10 configuracoes.

Agrega em 4 abas:
- Ranking: barras horizontais por NSE 6h
- NSE por horizonte: heatmap modelo x horizonte (1h, 3h, 6h, 12h, 24h)
- Forecasting x Continua: scatter do trade-off
- Recomendacoes: tabela de configuracao por aplicacao operacional

Consumido pelo StoryMap ArcGIS na secao de resultados. URL: /resultados_ablacao
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Resultados — Fase 1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Labels curtos e mapeamento
# ---------------------------------------------------------------------------
SHORT_LABELS = {
    "LSTM_Lumped": "Lumped",
    "LSTM": "LSTM puro",
    "LSTM_TTD_Base": "Base",
    "LSTM_TTD_Base_Fixed": "Base Fixed",
    "LSTM_TTD_Manning": "Manning",
    "LSTM_TTD_Manning_Fixed": "Manning Fixed",
    "LSTM_TTD_Base_SCS": "Base + SCS",
    "LSTM_TTD_Base_SCS_Fixed": "Base + SCS Fixed",
    "LSTM_TTD_Manning_SCS": "Manning + SCS",
    "LSTM_TTD_Manning_SCS_Fixed": "Manning + SCS Fixed",
}

HIGHLIGHTS = {
    "LSTM_TTD_Base": "Maior NSE em previsão 6h",
    "LSTM_TTD_Base_Fixed": "Maior NSE em simulação contínua",
    "LSTM_TTD_Manning": "NSE ≥ 0,81 em ambos os modos",
}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_forecast() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "summary.csv")
    df["label_curto"] = df["Modelo"].map(SHORT_LABELS).fillna(df["Modelo"])
    return df


@st.cache_data(show_spinner=False)
def _load_continuous() -> pd.DataFrame:
    with open(DATA_DIR / "summary_continuous.json", encoding="utf-8") as f:
        raw = json.load(f)
    rows = [
        {
            "Modelo": entry["model_name"],
            "NSE_cont": entry["metrics"]["nse"],
        }
        for entry in raw
    ]
    df = pd.DataFrame(rows)
    df["label_curto"] = df["Modelo"].map(SHORT_LABELS).fillna(df["Modelo"])
    return df


forecast = _load_forecast()
continuous = _load_continuous()
merged = forecast.merge(continuous[["Modelo", "NSE_cont"]], on="Modelo", how="inner")


# ---------------------------------------------------------------------------
# Funcoes de plotagem (adaptadas do app principal)
# ---------------------------------------------------------------------------
def make_nse_ranking(df: pd.DataFrame, horizon: str = "NSE_6h") -> go.Figure:
    """Barras horizontais por NSE no horizonte escolhido."""
    d = df.sort_values(horizon, ascending=True)
    fig = go.Figure(go.Bar(
        x=d[horizon],
        y=d["label_curto"],
        orientation="h",
        marker=dict(
            color=d[horizon],
            colorscale=[[0, "#fee2e2"], [0.5, "#fef3c7"], [1, "#15803d"]],
            cmin=0.5, cmax=0.85,
            showscale=False,
        ),
        text=d[horizon].round(3),
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>NSE: %{x:.3f}<extra></extra>",
    ))
    # Linha de corte Moriasi (NSE = 0,75)
    fig.add_vline(
        x=0.75, line_dash="dot", line_color="#16a34a", line_width=1.5,
        annotation_text="Muito Bom (≥ 0,75)",
        annotation_position="top right",
        annotation_font_color="#166534",
        annotation_font_size=10,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            title=f"NSE — {horizon.replace('NSE_', 'previsão ')}",
            range=[0.4, 0.92],
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(title="", gridcolor="#f1f5f9"),
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="white",
    )
    return fig


def make_nse_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap NSE por modelo x horizonte, com verde para NSE >= 0,75."""
    horizons = ["NSE_1h", "NSE_3h", "NSE_6h", "NSE_12h", "NSE_24h"]
    # Ordena por NSE_6h decrescente para leitura de cima para baixo
    d = df.sort_values("NSE_6h", ascending=False)
    z = d[horizons].values
    y = d["label_curto"].values
    x = ["1h", "3h", "6h", "12h", "24h"]

    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[
            [0.0, "#fee2e2"],   # NSE = 0,50 (vermelho claro)
            [0.50, "#fef3c7"],  # NSE ~ 0,68 (amarelo)
            [0.71, "#bbf7d0"],  # NSE ~ 0,75 (verde claro — Muito Bom)
            [1.0, "#15803d"],   # NSE = 0,85 (verde escuro)
        ],
        zmin=0.5, zmax=0.85,
        colorbar=dict(title="NSE", thickness=12, tickfont=dict(size=10)),
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, color="#0f172a"),
        hovertemplate="<b>%{y}</b><br>Horizonte: %{x}<br>NSE: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Horizonte de previsão", side="bottom"),
        yaxis=dict(title="", autorange="reversed"),
        height=480,
        margin=dict(l=20, r=20, t=30, b=40),
    )
    return fig


def make_tradeoff_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter NSE 6h (forecasting) x NSE (simulacao continua)."""
    fig = go.Figure()
    lims = [0.40, 0.90]

    # Diagonal y = x
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="#cbd5e1", dash="dash", width=1.5),
        name="y = x",
        hoverinfo="skip", showlegend=False,
    ))

    # Pontos
    for _, row in df.iterrows():
        is_highlight = row["Modelo"] in HIGHLIGHTS
        fig.add_trace(go.Scatter(
            x=[row["NSE_6h"]], y=[row["NSE_cont"]],
            mode="markers+text",
            text=[row["label_curto"]],
            textposition="top center",
            textfont=dict(
                size=11 if is_highlight else 10,
                color="#0f172a" if is_highlight else "#475569",
            ),
            marker=dict(
                size=20 if is_highlight else 12,
                color="#16a34a" if is_highlight else "#2563eb",
                line=dict(color="black", width=1.5 if is_highlight else 0.6),
                symbol="star" if is_highlight else "circle",
            ),
            name=row["label_curto"],
            customdata=[[row["Modelo"], row["NSE_6h"], row["NSE_cont"],
                         HIGHLIGHTS.get(row["Modelo"], "—")]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "NSE previsão 6h: %{customdata[1]:.3f}<br>"
                "NSE simulação contínua: %{customdata[2]:.3f}<br>"
                "Destaque: %{customdata[3]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # Linhas de corte NSE = 0,75 (classe Muito Bom, Moriasi 2007)
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
            title="<b>NSE — previsão 6 h</b>",
            range=lims, gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="<b>NSE — simulação contínua</b>",
            range=lims, gridcolor="#f1f5f9",
        ),
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
    )
    return fig


# ---------------------------------------------------------------------------
# Cabecalho minimo
# ---------------------------------------------------------------------------
st.markdown(
    "##### Estudo comparativo das 10 configurações — bacia do Rio Preto (Manuel Duarte)"
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rank, tab_heat, tab_trade, tab_rec = st.tabs([
    "Ranking", "NSE por horizonte", "Forecasting × Contínua", "Recomendações"
])

with tab_rank:
    fig = make_nse_ranking(forecast, horizon="NSE_6h")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Ranking das 10 configurações por NSE na previsão 6 h. "
        "Os resultados indicam que configurações com módulo TTD e parâmetros "
        "ajustáveis apresentam NSE superior em horizontes curtos — `LSTM_TTD_Base` "
        "(NSE = 0,837) lidera, seguida de `LSTM_TTD_Manning` (NSE = 0,830). "
        "A linha tracejada marca o limiar NSE = 0,75 (classe *Muito Bom* de Moriasi, 2007)."
    )

with tab_heat:
    fig = make_nse_heatmap(forecast)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Desempenho (NSE) por modelo e horizonte de previsão. A escala de cor "
        "destaca o limiar NSE ≥ 0,75 em tons de verde. Os resultados indicam "
        "dois padrões: `LSTM_TTD_Base` apresenta pico em 6 h e decaimento em "
        "horizontes longos; `LSTM_TTD_Manning_SCS` exibe melhora progressiva, "
        "atingindo NSE = 0,828 em 24 h."
    )

with tab_trade:
    fig = make_tradeoff_scatter(merged)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Diagonal tracejada: equivalência entre os dois modos de avaliação. "
        "Linhas pontilhadas: limiar NSE = 0,75. Configurações com parâmetros "
        "ajustáveis apresentam NSE superior em previsão de curto prazo; "
        "configurações com parâmetros fixos apresentam NSE superior em "
        "simulação contínua. As três estrelas verdes correspondem aos "
        "resultados pareto-ótimos em cada dimensão."
    )

with tab_rec:
    st.markdown("##### Recomendação de configuração por aplicação")
    st.markdown(
        """
| Aplicação | Configuração recomendada | NSE (6 h) | NSE (contínua) | Justificativa |
|---|---|---|---|---|
| **Alerta de cheias** | `LSTM_TTD_Base` (ajustável) | **0,837** | 0,697 | Maior acurácia em previsão de curto prazo; retreinamento periódico viabiliza parâmetros ajustáveis. |
| **Disponibilidade hídrica** | `LSTM_TTD_Base_Fixed` | 0,753 | **0,824** | Parâmetros físicos fixos atuam como regularização, garantindo estabilidade em séries longas. |
| **Uso robusto (ambos)** | `LSTM_TTD_Manning` | 0,830 | **0,809** | Degradação mínima (0,02 de NSE) entre os dois modos; rugosidade por uso do solo confere consistência. |
"""
    )
    st.caption(
        "A escolha entre parâmetros fixos e ajustáveis não decorre de "
        "superioridade técnica absoluta, mas de adequação ao regime operacional. "
        "Configurações com parâmetros ajustáveis apresentam NSE superior em "
        "previsão de curto prazo; configurações com parâmetros fixos apresentam "
        "NSE superior em simulação contínua."
    )
