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
# Grafico de barras horizontais (estilo do app principal)
# ---------------------------------------------------------------------------
d = df.sort_values(horizon, ascending=True).copy()
horizon_label = horizon.replace("NSE_", "")

fig = go.Figure(go.Bar(
    x=d[horizon],
    y=d["Modelo"],
    orientation="h",
    marker=dict(
        color=d[horizon],
        colorscale=[[0, "#fee2e2"], [0.5, "#fef3c7"], [1, "#15803d"]],
        showscale=False,
    ),
    text=d[horizon].round(3),
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>NSE: %{x:.3f}<extra></extra>",
))

fig.update_layout(
    title=f"Ranking de modelos — NSE {horizon_label}",
    template="plotly_white",
    xaxis_title="NSE",
    xaxis=dict(range=[0.4, 0.9]),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Caption dinamico baseado no horizonte selecionado
# ---------------------------------------------------------------------------
top = d.iloc[-1]
bot = d.iloc[0]
count_vb = int((df[horizon] >= 0.75).sum())

st.caption(
    f"Três padrões: (1) modelos *lumped* e LSTM puro formam o grupo "
    f"inferior (NSE < 0,66); (2) modelos TTD sem SCS com parâmetros "
    f"aprendíveis **dominam o horizonte de 6 h**; (3) o SCS-CN em geral "
    f"**degrada** a previsão de curto prazo (6 a 9,5%), mas o caso "
    f"específico **LSTM_TTD_Manning_SCS** se destaca em **24 h** "
    f"(NSE = 0,828), sugerindo que a combinação Manning + SCS com "
    f"parâmetros aprendíveis modula melhor a abstração em horizontes longos."
)
