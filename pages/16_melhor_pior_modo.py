"""
Widget: melhor e pior configuracao em cada modo de avaliacao.

Recorte didatico do estudo comparativo da Fase 1:
- Previsao 6 h: melhor (LSTM_TTD_Base) vs pior (LSTM_Lumped, que tambem e
  a arquitetura mais simples)
- Simulacao continua: melhor (LSTM_TTD_Base_Fixed) vs baseline simples
  (LSTM_Lumped) vs pior (LSTM_TTD_Manning_Fixed)

Destaque na continua: o LSTM_Lumped (sem fisica, concentrado) supera o
Manning_Fixed (com fisica, mas com Tc fixo em escala 1) — fisica mal
calibrada pode ser pior que fisica ausente.

Consumido pelo StoryMap ArcGIS. URL: /melhor_pior_modo
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Melhor × Pior — TTD-SCS-LSTM",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Labels curtos
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


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    forecast = pd.read_csv(DATA_DIR / "summary.csv")
    with open(DATA_DIR / "summary_continuous.json", encoding="utf-8") as f:
        cont_raw = json.load(f)
    continuous = pd.DataFrame(
        [{"Modelo": m["model_name"], "NSE_cont": m["metrics"]["nse"]} for m in cont_raw]
    )
    df = forecast.merge(continuous, on="Modelo", how="inner")
    df["label_curto"] = df["Modelo"].map(SHORT_LABELS).fillna(df["Modelo"])
    return df


df = _load()

# Extremos e referencia simples
melhor_f = df.loc[df["NSE_6h"].idxmax()]
pior_f = df.loc[df["NSE_6h"].idxmin()]
melhor_c = df.loc[df["NSE_cont"].idxmax()]
pior_c = df.loc[df["NSE_cont"].idxmin()]
lumped = df.loc[df["Modelo"] == "LSTM_Lumped"].iloc[0]


# ---------------------------------------------------------------------------
# Classificacao Moriasi (2007)
# ---------------------------------------------------------------------------
def classe_moriasi(nse: float) -> str:
    if nse >= 0.75:
        return "Muito Bom"
    if nse >= 0.65:
        return "Bom"
    if nse >= 0.50:
        return "Satisfatório"
    return "Insatisfatório"


# ---------------------------------------------------------------------------
# Cabecalho minimo
# ---------------------------------------------------------------------------
st.markdown(
    "##### Maior e menor NSE em cada modo de avaliação — estudo comparativo da Fase 1"
)


# ---------------------------------------------------------------------------
# Figura: subplot com 2 paineis (Forecast | Continua)
# ---------------------------------------------------------------------------
COR_MELHOR = "#16a34a"   # verde
COR_BASELINE = "#94a3b8" # cinza ardosia (referencia simples)
COR_PIOR = "#dc2626"     # vermelho
COR_GRID = "#f1f5f9"

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Previsão 6 horas à frente</b>",
        "<b>Simulação contínua (9 meses)</b>",
    ),
    horizontal_spacing=0.20,
)

# --- Painel 1: Previsao 6 h (2 barras — no forecast, o Lumped ja e o pior) ---
y_f = [
    f"Pior / baseline simples<br><span style='font-size:11px;color:#64748b;'>{pior_f['label_curto']} (concentrado)</span>",
    f"Melhor configuração<br><span style='font-size:11px;color:#64748b;'>{melhor_f['label_curto']}</span>",
]
x_f = [pior_f["NSE_6h"], melhor_f["NSE_6h"]]
classes_f = [classe_moriasi(x_f[0]), classe_moriasi(x_f[1])]
cores_f = [COR_PIOR, COR_MELHOR]

fig.add_trace(go.Bar(
    x=x_f, y=y_f,
    orientation="h",
    marker=dict(color=cores_f, line=dict(color="black", width=0.8)),
    text=[f"<b>{v:.3f}</b>  ({c})" for v, c in zip(x_f, classes_f)],
    textposition="outside",
    textfont=dict(size=13, color="#0f172a"),
    cliponaxis=False,
    hovertemplate="NSE: %{x:.3f}<extra></extra>",
    showlegend=False,
), row=1, col=1)

# --- Painel 2: Simulacao continua (3 barras — melhor, baseline Lumped, pior) ---
y_c = [
    f"Pior configuração<br><span style='font-size:11px;color:#64748b;'>{pior_c['label_curto']} (Tc fixo em escala 1)</span>",
    f"Baseline simples<br><span style='font-size:11px;color:#64748b;'>{lumped['label_curto']} (concentrado, sem física)</span>",
    f"Melhor configuração<br><span style='font-size:11px;color:#64748b;'>{melhor_c['label_curto']}</span>",
]
x_c = [pior_c["NSE_cont"], lumped["NSE_cont"], melhor_c["NSE_cont"]]
classes_c = [classe_moriasi(v) for v in x_c]
cores_c = [COR_PIOR, COR_BASELINE, COR_MELHOR]

fig.add_trace(go.Bar(
    x=x_c, y=y_c,
    orientation="h",
    marker=dict(color=cores_c, line=dict(color="black", width=0.8)),
    text=[f"<b>{v:.3f}</b>  ({c})" for v, c in zip(x_c, classes_c)],
    textposition="outside",
    textfont=dict(size=13, color="#0f172a"),
    cliponaxis=False,
    hovertemplate="NSE: %{x:.3f}<extra></extra>",
    showlegend=False,
), row=1, col=2)

# Linha de corte Moriasi NSE = 0,75 (Muito Bom) em cada painel
for col in (1, 2):
    fig.add_vline(
        x=0.75, line_dash="dot", line_color="#16a34a", line_width=1.5,
        row=1, col=col,
    )

# Anotacao sinalizando a linha de corte (so no painel 1)
fig.add_annotation(
    x=0.75, y=1.10, xref="x", yref="paper",
    text="<span style='color:#166534;font-size:10px;'>NSE = 0,75 → Muito Bom (Moriasi, 2007)</span>",
    showarrow=False, xanchor="center",
    row=1, col=1,
)

# Eixos
for col in (1, 2):
    fig.update_xaxes(
        range=[0.0, 1.0],
        gridcolor=COR_GRID,
        zeroline=False,
        title_text="NSE",
        title_font=dict(size=11),
        row=1, col=col,
    )
    fig.update_yaxes(
        gridcolor=COR_GRID,
        tickfont=dict(size=12),
        row=1, col=col,
    )

fig.update_layout(
    template="plotly_white",
    height=420,
    margin=dict(l=20, r=40, t=70, b=40),
    plot_bgcolor="white",
    bargap=0.40,
)

# Subtitulos dos paineis em fonte maior
for ann in fig.layout.annotations[:2]:
    ann.font = dict(size=14, color="#0f172a")

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Caption — leitura dos achados
# ---------------------------------------------------------------------------
gap_f = melhor_f["NSE_6h"] - pior_f["NSE_6h"]
gap_c = melhor_c["NSE_cont"] - pior_c["NSE_cont"]

st.caption(
    f"**Leitura.** Em cada modo, a diferença de NSE entre a melhor e a pior "
    f"configuração é expressiva (Δ = {gap_f:.2f} na previsão 6 h, "
    f"Δ = {gap_c:.2f} na contínua), evidenciando que as escolhas de "
    f"arquitetura e parametrização têm efeito de mesma ordem de grandeza "
    f"em ambas as aplicações. Em previsão 6 h, o `LSTM_Lumped` já corresponde "
    f"à pior configuração — a ausência de discretização espacial penaliza "
    f"diretamente a acurácia de curto prazo. Em simulação contínua, o cenário "
    f"se inverte: o `LSTM_Lumped` (baseline simples, NSE = {lumped['NSE_cont']:.3f}) "
    f"**supera** o `Manning_Fixed` (NSE = {pior_c['NSE_cont']:.3f}), indicando que "
    f"uma arquitetura com componente físico porém sem calibração do fator de "
    f"escala do tempo de concentração pode apresentar desempenho inferior ao "
    f"de um modelo concentrado sem física explícita. As três configurações "
    f"com NSE mais alto (melhor em cada modo e o baseline simples na contínua) "
    f"permanecem iguais ou acima da classe *Satisfatório* de Moriasi (2007)."
)
