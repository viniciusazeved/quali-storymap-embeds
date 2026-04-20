"""
Widget: ranking das 10 configuracoes por NSE, com slider de horizonte.

Replica a aba "Ranking" da pagina /resultados do app principal:
barras horizontais ordenadas pelo NSE do horizonte selecionado
(1 h, 3 h, 6 h, 12 h ou 24 h), com escala de cor e linha de corte
Moriasi (NSE = 0,75).

Mostra como o ranking se reordena com o horizonte — em particular,
`LSTM_TTD_Base` domina 1–6 h e `LSTM_TTD_Manning_SCS` desponta em 24 h.

URL: /ranking_nse
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Ranking NSE — TTD-SCS-LSTM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"

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

HORIZON_OPTIONS = ["NSE_1h", "NSE_3h", "NSE_6h", "NSE_12h", "NSE_24h"]


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "summary.csv")
    df["label_curto"] = df["Modelo"].map(SHORT_LABELS).fillna(df["Modelo"])
    return df


df = _load()

# ---------------------------------------------------------------------------
# Controle — slider de horizonte
# ---------------------------------------------------------------------------
horizon = st.select_slider(
    "Horizonte de previsão",
    options=HORIZON_OPTIONS,
    value="NSE_6h",
    format_func=lambda x: x.replace("NSE_", ""),
)

# ---------------------------------------------------------------------------
# Grafico de barras horizontais
# ---------------------------------------------------------------------------
d = df.sort_values(horizon, ascending=True).copy()

fig = go.Figure(go.Bar(
    x=d[horizon],
    y=d["label_curto"],
    orientation="h",
    marker=dict(
        color=d[horizon],
        colorscale=[[0, "#fee2e2"], [0.5, "#fef3c7"], [1, "#15803d"]],
        cmin=0.5, cmax=0.85,
        showscale=False,
        line=dict(color="black", width=0.6),
    ),
    text=[f"<b>{v:.3f}</b>" for v in d[horizon]],
    textposition="outside",
    textfont=dict(size=12, color="#0f172a"),
    cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>NSE: %{x:.3f}<extra></extra>",
))

# Linha de corte Moriasi (NSE = 0,75)
fig.add_vline(
    x=0.75, line_dash="dot", line_color="#16a34a", line_width=1.5,
    annotation_text="Muito Bom (Moriasi, 2007)",
    annotation_position="top right",
    annotation_font_color="#166534",
    annotation_font_size=10,
)

# Linha de corte Moriasi inferior (NSE = 0,50, Satisfatorio)
fig.add_vline(
    x=0.50, line_dash="dot", line_color="#94a3b8", line_width=1,
    annotation_text="Satisfatório ≥ 0,50",
    annotation_position="bottom right",
    annotation_font_color="#64748b",
    annotation_font_size=9,
)

horizon_label = horizon.replace("NSE_", "")
fig.update_layout(
    template="plotly_white",
    xaxis=dict(
        title=f"<b>NSE — previsão {horizon_label} à frente</b>",
        range=[0.40, 0.92],
        gridcolor="#f1f5f9",
    ),
    yaxis=dict(title="", gridcolor="#f1f5f9", tickfont=dict(size=12)),
    height=490,
    margin=dict(l=20, r=40, t=20, b=40),
    plot_bgcolor="white",
    bargap=0.25,
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Caption dinamico baseado no horizonte selecionado
# ---------------------------------------------------------------------------
top = d.iloc[-1]
bot = d.iloc[0]
count_vb = int((df[horizon] >= 0.75).sum())

st.caption(
    f"**Horizonte {horizon_label}** — Líder: `{top['Modelo']}` "
    f"(NSE = {top[horizon]:.3f}). Último: `{bot['Modelo']}` "
    f"(NSE = {bot[horizon]:.3f}). "
    f"{count_vb} de {len(df)} configurações classificam-se como "
    f"*Muito Bom* (NSE ≥ 0,75) neste horizonte. "
    f"O ranking se reordena em função do horizonte: o `LSTM_TTD_Base` "
    f"domina horizontes de 1 a 6 h, enquanto o `LSTM_TTD_Manning_SCS` "
    f"desponta em 24 h (NSE = 0,828), sugerindo que a combinação de "
    f"rugosidade por uso do solo e SCS-CN com parâmetros aprendíveis "
    f"modula melhor a resposta em horizontes longos."
)
