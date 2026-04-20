"""
Widget: parametros fisicos aprendidos pelos modelos com componente TTD.

Exibe os valores convergidos de t_c_scale (fator de escala do tempo de
concentracao), sigma (dispersao do IUH gaussiano) e lambda_scs (coeficiente
de abstracao inicial) para cada configuracao com parametros aprendiveis.

Destaque: o t_c_scale converge para ~1,2-1,3 (bacia responde 20-30 % mais
lentamente que Maidment); lambda converge para 0,06-0,17, consistente com
ValleJunior et al. (2019) para bacias tropicais brasileiras.

URL: /parametros_aprendidos
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
    page_title="Parâmetros aprendidos — TTD-SCS-LSTM",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"

SHORT_LABELS = {
    "LSTM_TTD_Base": "Base",
    "LSTM_TTD_Manning": "Manning",
    "LSTM_TTD_Base_SCS": "Base + SCS",
    "LSTM_TTD_Manning_SCS": "Manning + SCS",
}


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    with open(DATA_DIR / "all_results.json", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for entry in raw:
        lp = entry.get("learned_params") or {}
        if not lp:
            continue
        tc = lp.get("tc_scale")
        # Pula configuracoes Fixed (tc_scale = 1.0 exato)
        if tc is None or abs(tc - 1.0) < 1e-6:
            continue
        rows.append({
            "Modelo": entry["model_name"],
            "label_curto": SHORT_LABELS.get(entry["model_name"], entry["model_name"]),
            "tc_scale": float(lp["tc_scale"]),
            "sigma": float(lp["sigma"]),
            "lambda_scs": (
                float(lp["lambda_scs"]) if lp.get("lambda_scs") is not None else None
            ),
            "nse_6h": float(entry["test_by_horizon"]["6h"]["nse"]),
            "epochs_trained": int(entry.get("epochs_trained", 0)),
        })
    return pd.DataFrame(rows).sort_values("nse_6h", ascending=False).reset_index(drop=True)


df = _load()

st.markdown(
    "##### Parâmetros físicos aprendidos — configurações com parâmetros ajustáveis"
)

st.caption(
    "Valores convergidos ao fim do treinamento nos modelos com componente "
    "TTD e parâmetros aprendíveis (excluem configurações *Fixed*). "
    "Os valores de referência da literatura aparecem como linhas pontilhadas."
)

# ---------------------------------------------------------------------------
# Figura com 3 paineis (tc_scale, sigma, lambda)
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "<b>t<sub>c,scale</sub> (α)</b><br><sup>fator de escala do T<sub>c</sub></sup>",
        "<b>σ</b><br><sup>dispersão do IUH (h)</sup>",
        "<b>λ<sub>SCS</sub></b><br><sup>abstração inicial</sup>",
    ),
    horizontal_spacing=0.11,
)

cores = ["#16a34a", "#0891b2", "#d97706", "#7c3aed"][:len(df)]

# Painel 1 — tc_scale
fig.add_trace(go.Bar(
    x=df["tc_scale"], y=df["label_curto"],
    orientation="h",
    marker=dict(color=cores, line=dict(color="black", width=0.6)),
    text=[f"<b>{v:.2f}</b>" for v in df["tc_scale"]],
    textposition="outside",
    textfont=dict(size=11),
    cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>t_c,scale = %{x:.3f}<extra></extra>",
    showlegend=False,
), row=1, col=1)
# Referencia tc_scale = 1.0 (Maidment)
fig.add_vline(x=1.0, line_dash="dot", line_color="#64748b",
              line_width=1, row=1, col=1)

# Painel 2 — sigma
fig.add_trace(go.Bar(
    x=df["sigma"], y=df["label_curto"],
    orientation="h",
    marker=dict(color=cores, line=dict(color="black", width=0.6)),
    text=[f"<b>{v:.2f}</b> h" for v in df["sigma"]],
    textposition="outside",
    textfont=dict(size=11),
    cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>σ = %{x:.3f} h<extra></extra>",
    showlegend=False,
), row=1, col=2)
# Referencia sigma = 3.0 (valor fixo das configuracoes Fixed)
fig.add_vline(x=3.0, line_dash="dot", line_color="#64748b",
              line_width=1, row=1, col=2)

# Painel 3 — lambda (so para modelos com SCS)
df_lambda = df[df["lambda_scs"].notna()].copy()
if not df_lambda.empty:
    # Paleta restrita aos modelos com SCS
    cores_lambda = [cores[df.index[df["Modelo"] == m][0]] for m in df_lambda["Modelo"]]
    fig.add_trace(go.Bar(
        x=df_lambda["lambda_scs"], y=df_lambda["label_curto"],
        orientation="h",
        marker=dict(color=cores_lambda, line=dict(color="black", width=0.6)),
        text=[f"<b>{v:.3f}</b>" for v in df_lambda["lambda_scs"]],
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>λ = %{x:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=3)
    # Referencia lambda = 0.20 (padrao USDA)
    fig.add_vline(x=0.20, line_dash="dot", line_color="#64748b",
                  line_width=1, row=1, col=3,
                  annotation_text="λ = 0,20 (USDA)",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="#64748b"))
    # Referencia ValleJunior 2019 = 0.045 (mediana bacias tropicais BR)
    fig.add_vline(x=0.045, line_dash="dot", line_color="#16a34a",
                  line_width=1, row=1, col=3,
                  annotation_text="λ = 0,045 (BR)",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#166534"))

# Eixos
fig.update_xaxes(title_text="α", range=[0, max(df["tc_scale"]) * 1.25], row=1, col=1,
                 gridcolor="#f1f5f9")
fig.update_xaxes(title_text="σ (h)", range=[0, max(df["sigma"]) * 1.2], row=1, col=2,
                 gridcolor="#f1f5f9")
fig.update_xaxes(title_text="λ", range=[0, 0.28], row=1, col=3,
                 gridcolor="#f1f5f9")
for col in (1, 2, 3):
    fig.update_yaxes(gridcolor="#f1f5f9", tickfont=dict(size=11), row=1, col=col)

fig.update_layout(
    template="plotly_white",
    height=360,
    margin=dict(l=20, r=20, t=70, b=40),
    plot_bgcolor="white",
    bargap=0.35,
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Interpretacao fisica
# ---------------------------------------------------------------------------
st.markdown(
    """
**Interpretação física.** Os parâmetros convergidos têm leitura direta do
ponto de vista hidrológico:

- **t<sub>c,scale</sub> ≈ 1,2–1,3** para modelos Base e Manning: a bacia **responde 20–30 % mais
  lentamente** que o estimado pelo método de Maidment, possivelmente
  indicando efeito de armazenamento temporário em planícies de inundação
  ou subestimativa sistemática do método clássico para bacias deste porte
  (3.117 km²).

- **σ ≈ 4–5 h** (Base, Manning): hidrogramas com dispersão maior que o
  valor fixo de referência (3 h). Em configurações com SCS-CN, σ cresce
  ainda mais (~12–14 h), refletindo a combinação da abstração inicial
  com a propagação distribuída.

- **λ convergiu para 0,06–0,17** — consistente com ValleJunior *et al.*
  (2019) para bacias tropicais brasileiras (mediana 0,045) e
  **substancialmente inferior** ao padrão 0,20 de condições
  norte-americanas (Woodward *et al.*, 2003). Isso reforça a necessidade
  de calibração regional do SCS-CN para aplicações no Brasil.
    """,
    unsafe_allow_html=True,
)

# Tabela compacta
st.markdown("**Tabela de parâmetros aprendidos**")
st.dataframe(
    df.rename(columns={
        "Modelo": "Modelo",
        "tc_scale": "t_c,scale",
        "sigma": "σ (h)",
        "lambda_scs": "λ",
        "nse_6h": "NSE 6 h",
        "epochs_trained": "Épocas",
    }).drop(columns=["label_curto"]).style.format({
        "t_c,scale": "{:.3f}",
        "σ (h)": "{:.2f}",
        "λ": "{:.3f}",
        "NSE 6 h": "{:.3f}",
        "Épocas": "{:.0f}",
    }, na_rep="—"),
    use_container_width=True, hide_index=True,
)
