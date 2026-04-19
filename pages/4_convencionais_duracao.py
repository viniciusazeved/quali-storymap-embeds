"""
Widget: distribuicao da duracao das series das estacoes convencionais da ANA.

Replicado a partir do app principal (pagina "3. Monitoramento convencional",
grafico plot_ana_nyd_distribution), adaptado para embed no StoryMap.

Fonte: catalogo ANAF (Carvalho & Braga, 2020).
URL: /convencionais_duracao
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Duração das séries — convencionais",
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
        "Operadora",
        ["Todas", "Apenas ANA"],
        index=0,
        horizontal=False,
    )
only_ana = "ANA" in recorte

with c2:
    st.caption(
        "Distribuição da duração das séries fluviométricas convencionais no "
        "catálogo ANAF. Cada barra representa a quantidade de estações cuja "
        "série registrada tem a duração indicada em anos."
    )


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def _load(only_ana: bool) -> pd.DataFrame:
    fname = "estacoes_convencionais_ana.csv" if only_ana else "estacoes_convencionais_todas.csv"
    return pd.read_csv(DATA_DIR / fname)


df = _load(only_ana)
nyd = df["NYD"].dropna().astype(float)

# ---------------------------------------------------------------------------
# Histograma
# ---------------------------------------------------------------------------
fig = go.Figure(go.Histogram(
    x=nyd,
    nbinsx=50,
    marker_color="#2563eb",
    opacity=0.85,
    name="Estações",
    hovertemplate="Duração: %{x:.0f} anos<br>Estações: %{y}<extra></extra>",
))

# Linhas de referência em 20 e 40 anos
fig.add_vline(
    x=20, line_dash="dash", line_color="#16a34a",
    annotation_text="20 anos (mínimo recomendado)",
    annotation_position="top right",
    annotation_font=dict(size=10, color="#166534"),
)
fig.add_vline(
    x=40, line_dash="dot", line_color="#94a3b8",
    annotation_text="40 anos",
    annotation_position="top right",
    annotation_font=dict(size=10, color="#475569"),
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="<b>Duração da série (anos)</b>",
    yaxis_title="<b>Número de estações</b>",
    height=460,
    margin=dict(l=40, r=30, t=20, b=40),
    bargap=0.05,
)

st.plotly_chart(fig, use_container_width=True)

# KPIs resumidos abaixo do gráfico
k1, k2, k3, k4 = st.columns(4)
k1.metric("Estações no recorte", f"{len(df):,}".replace(",", "."))
k2.metric("Com ≥ 20 anos", f"{int((nyd >= 20).sum()):,}".replace(",", "."))
k3.metric("Com ≥ 40 anos", f"{int((nyd >= 40).sum()):,}".replace(",", "."))
k4.metric("Mediana (anos)", f"{nyd.median():.0f}")

st.caption(
    f"**Fonte:** catálogo ANAF (Carvalho & Braga, 2020) — {len(df):,} estações no recorte "
    f"**{'Apenas ANA' if only_ana else 'Todas as operadoras'}**. "
    "NYD = *number of years of data* (anos com dado registrado no HidroWeb)."
    .replace(",", ".")
)
