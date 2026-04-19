"""
Widget: busca preliminar de hiperparametros na Fase 1.

Replica a analise exploratoria disponivel no app principal: apresenta os
resultados da busca em grade (3 modelos x lookback x hidden_size x num_layers),
com coordenadas paralelas, scatter n_params x NSE e um resumo das tres
melhores configuracoes por modelo.

Dados: `storymap_embeds/data/hyperparam_search.csv` — copia de
`streamlit_apresentacao/data/outputs/hyperparam_search/search_20260124_202544/results.csv`.
Busca preliminar: 150 epocas, paciencia 20, 108 configuracoes treinadas.

Consumido pelo StoryMap ArcGIS. URL: /hiperparametros
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Busca de hiperparâmetros",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"

# Rotulos amigaveis para exibicao
MODEL_LABELS = {
    "lstm": "LSTM",
    "lstm_ttd_aprend": "LSTM_TTD_Aprend",
    "lstm_ttd_baseflow": "LSTM_TTD_Baseflow",
}


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "hyperparam_search.csv")
    df["model_label"] = df["model_type"].map(MODEL_LABELS).fillna(df["model_type"])
    return df


df = _load()

# ---------------------------------------------------------------------------
# Cabecalho minimo
# ---------------------------------------------------------------------------
st.markdown(
    "##### Busca preliminar de hiperparâmetros — Fase 1"
)
st.caption(
    f"Busca em grade com {len(df)} configurações, treinamento reduzido "
    f"(150 épocas, paciência 20, semente 42). Três variantes arquiteturais "
    f"do TTD-SCS-LSTM foram exploradas variando janela de histórico, "
    f"tamanho da camada oculta e número de camadas LSTM."
)

tab_par, tab_scatter, tab_best = st.tabs([
    "Coordenadas paralelas",
    "Parâmetros × NSE",
    "Melhores configurações",
])

# ---------------------------------------------------------------------------
# Tab 1: coordenadas paralelas
# ---------------------------------------------------------------------------
with tab_par:
    cols_num = ["lookback", "hidden_size", "num_layers", "n_params", "test_nse"]
    df_clean = df.dropna(subset=cols_num + ["model_type"]).copy()

    mt_codes, mt_labels = pd.factorize(df_clean["model_type"])
    mt_ticktext = [MODEL_LABELS.get(m, m) for m in mt_labels]

    dims = [
        dict(
            label="Modelo",
            values=mt_codes,
            tickvals=list(range(len(mt_labels))),
            ticktext=mt_ticktext,
        ),
        dict(label="Lookback (h)", values=df_clean["lookback"]),
        dict(label="Hidden size", values=df_clean["hidden_size"]),
        dict(label="N. camadas", values=df_clean["num_layers"]),
        dict(label="N. parâmetros", values=df_clean["n_params"]),
        dict(label="NSE (teste)", values=df_clean["test_nse"]),
    ]

    fig_par = go.Figure(go.Parcoords(
        line=dict(
            color=df_clean["test_nse"],
            colorscale="Viridis",
            showscale=True,
            cmin=df_clean["test_nse"].min(),
            cmax=df_clean["test_nse"].max(),
            colorbar=dict(title="NSE"),
        ),
        dimensions=dims,
    ))
    fig_par.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=60, r=40, t=40, b=20),
    )

    st.plotly_chart(fig_par, use_container_width=True)

    st.caption(
        "Cada linha representa uma configuração treinada; a cor acompanha o "
        "NSE de teste. As linhas de topo, concentradas em tons amarelos, "
        "correspondem às configurações com maior capacidade preditiva — "
        "convergência visual nas faixas *hidden size* = 64–128 e "
        "3 camadas LSTM."
    )

# ---------------------------------------------------------------------------
# Tab 2: scatter n_params x test_nse
# ---------------------------------------------------------------------------
with tab_scatter:
    fig_sc = px.scatter(
        df,
        x="n_params",
        y="test_nse",
        color="model_label",
        symbol="num_layers",
        hover_data={
            "lookback": True,
            "hidden_size": True,
            "num_layers": True,
            "n_params": ":,",
            "test_nse": ":.3f",
            "model_label": False,
        },
        labels={
            "n_params": "Número de parâmetros treináveis",
            "test_nse": "NSE — conjunto de teste",
            "model_label": "Modelo",
            "num_layers": "Camadas LSTM",
        },
        opacity=0.85,
        color_discrete_sequence=["#2563eb", "#16a34a", "#dc2626"],
    )
    fig_sc.update_traces(marker=dict(size=11, line=dict(color="black", width=0.5)))
    fig_sc.update_layout(
        template="plotly_white",
        height=520,
        xaxis=dict(type="log", gridcolor="#f1f5f9"),
        yaxis=dict(range=[0.40, 0.85], gridcolor="#f1f5f9"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",
    )

    # Linha de referencia NSE = 0,75 (classe Muito Bom de Moriasi)
    fig_sc.add_hline(
        y=0.75, line=dict(color="#cbd5e1", dash="dot", width=1),
        annotation_text="NSE = 0,75 (<i>Muito Bom</i>, Moriasi 2007)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#475569"),
    )

    st.plotly_chart(fig_sc, use_container_width=True)

    st.caption(
        "Relação entre número de parâmetros treináveis (escala log) e NSE "
        "no conjunto de teste. O aumento de parâmetros, isoladamente, não "
        "traduz-se em desempenho monotônico — configurações com ~150 mil "
        "parâmetros chegam a superar redes com 400 mil, indicando que o "
        "balanço entre capacidade e regularização é mais determinante do "
        "que o tamanho absoluto da arquitetura."
    )

# ---------------------------------------------------------------------------
# Tab 3: melhores configuracoes por modelo
# ---------------------------------------------------------------------------
with tab_best:
    st.markdown(
        "##### Três melhores configurações por modelo (ordenadas por NSE de teste)"
    )

    top3 = (
        df.sort_values("test_nse", ascending=False)
          .groupby("model_label", group_keys=False)
          .head(3)
          .sort_values(["model_label", "test_nse"], ascending=[True, False])
          .reset_index(drop=True)
    )

    for model_label in ["LSTM", "LSTM_TTD_Aprend", "LSTM_TTD_Baseflow"]:
        sub = top3[top3["model_label"] == model_label]
        if sub.empty:
            continue
        st.markdown(f"**{model_label}**")
        display = sub[[
            "lookback", "hidden_size", "num_layers", "n_params",
            "test_nse", "val_nse", "epochs_trained",
        ]].copy()
        display.columns = [
            "Lookback (h)", "Hidden size", "N. camadas",
            "N. parâmetros", "NSE (teste)", "NSE (validação)", "Épocas",
        ]
        display["N. parâmetros"] = display["N. parâmetros"].map(lambda v: f"{int(v):,}".replace(",", "."))
        display["NSE (teste)"] = display["NSE (teste)"].map(lambda v: f"{v:.3f}")
        display["NSE (validação)"] = display["NSE (validação)"].map(lambda v: f"{v:.3f}")
        st.dataframe(display, hide_index=True, use_container_width=True)

    st.caption(
        "A busca preliminar indicou três sinais consistentes: (i) "
        "3 camadas LSTM superam 2 em todos os modelos; (ii) o lookback "
        "ótimo varia com a arquitetura — 120 h para `LSTM_TTD_Aprend` e "
        "240 h para `LSTM` e `LSTM_TTD_Baseflow`; (iii) o NSE máximo "
        "observado situa-se em torno de 0,80, próximo ao da configuração "
        "adotada na Fase 1. Esses resultados motivam uma busca refinada "
        "via otimização bayesiana (Optuna) na Fase 2 multi-bacia."
    )
