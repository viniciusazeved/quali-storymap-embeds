"""
Widget: panorama dos dados de entrada e saida utilizados no modelo.

Replicado a partir do app principal (pagina "8. Dados"), com foco em
exploracao interativa de vazao, precipitacao e atributos das ottobacias.

Dados exportados de dataset_v2.h5:
- serie_bacia.csv: vazao horaria e precipitacao media areal (2021-2025)
- ottobacias_attrs.csv: atributos das 245 ottobacias

URL: /dados
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
    page_title="Dados — TTD-SCS-LSTM",
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
def _load_serie() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "serie_bacia.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    return df


@st.cache_data(show_spinner=False)
def _load_attrs() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ottobacias_attrs.csv")


serie = _load_serie()
attrs = _load_attrs()

# ---------------------------------------------------------------------------
# Abertura
# ---------------------------------------------------------------------------
st.markdown(
    """
    Dados de entrada e referência utilizados na Fase 1 do modelo
    TTD-SCS-LSTM. A vazão observada provém da estação ANA 58585000
    (Manuel Duarte), a precipitação é produto MERGE/CPTEC (resolução
    espacial de ~10 km, horária), agregada por intersecção com as 245
    ottobacias. Os atributos fisiográficos são derivados do ANADEM 30 m
    (IPH/UFRGS + ANA), do produto BHAE_CN-2022 da ANA e do MapBiomas
    Coleção 8.0.
    """
)

# ---------------------------------------------------------------------------
# KPIs de resumo
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Período",
    f"{serie.index.min().year}–{serie.index.max().year}",
    f"{len(serie):,} horas".replace(",", "."),
)
k2.metric(
    "Vazão média",
    f"{serie['Q_obs'].mean():.1f} m³/s",
    f"máx {serie['Q_obs'].max():.0f} m³/s",
)
k3.metric(
    "Precipitação anual média",
    f"{serie['P_mean'].resample('YE').sum().mean():.0f} mm",
    help="Média espacial sobre as 245 ottobacias, totalizada anualmente.",
)
k4.metric(
    "Ottobacias",
    f"{len(attrs)}",
    f"área média {attrs['area_km2'].mean():.1f} km²",
)

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_serie, tab_precip, tab_perm, tab_saz, tab_disp, tab_otto = st.tabs([
    "Série de vazão",
    "Precipitação",
    "Curva de permanência",
    "Sazonalidade",
    "Disponibilidade",
    "Ottobacias",
])

# ========================================================= Série de vazão
with tab_serie:
    # Split visual: train/val/test em cores distintas
    fig = go.Figure()
    colors = {"train": "#93c5fd", "val": "#fde68a", "test": "#fca5a5"}
    labels = {"train": "Treinamento", "val": "Validação", "test": "Teste"}
    for split in ("train", "val", "test"):
        mask = serie["split"] == split
        if mask.any():
            fig.add_trace(go.Scatter(
                x=serie.index[mask], y=serie.loc[mask, "Q_obs"],
                mode="lines",
                line=dict(color=colors[split], width=1),
                name=labels[split],
            ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Tempo",
        yaxis_title="Q (m³/s)",
        height=440,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Série horária concatenada da estação ANA 58585000 (Manuel Duarte), "
        "com particionamento temporal em treinamento (~70%), validação (~15%) "
        "e teste (~15%). Valores derivados da telemetria ANA com tratamento "
        "para picos espúrios e lacunas."
    )

# =========================================================== Precipitação
with tab_precip:
    # Série horária
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=serie.index, y=serie["P_mean"],
        marker_color="#0ea5e9", opacity=0.8,
        name="Precipitação horária",
        hovertemplate="%{x}<br>P = %{y:.2f} mm<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Tempo",
        yaxis_title="P (mm/h)",
        height=340,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Totais anuais
    annual = serie["P_mean"].resample("YE").sum()
    fig_an = go.Figure(go.Bar(
        x=annual.index.year, y=annual.values,
        marker_color="#0ea5e9", opacity=0.85,
        text=[f"{v:.0f}" for v in annual.values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.0f} mm<extra></extra>",
    ))
    fig_an.add_hline(
        y=annual.mean(), line_dash="dash", line_color="#dc2626",
        annotation_text=f"Média = {annual.mean():.0f} mm",
        annotation_position="top right",
    )
    fig_an.update_layout(
        title="Precipitação anual (mm)",
        template="plotly_white",
        xaxis_title="Ano", yaxis_title="P (mm)",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_an, use_container_width=True)
    st.caption(
        "Produto MERGE do CPTEC/INPE (Rozante et al., 2010) — combinação de "
        "estimativas de satélite (TRMM/GPM) com observações de superfície "
        "(ANA, DAEE, SIMEPAR, CEMADEN, INEMA). Série média espacial sobre as "
        "245 ottobacias."
    )

# ======================================================== Permanência
with tab_perm:
    q = serie["Q_obs"].dropna().values
    sorted_q = np.sort(q)[::-1]
    perc = np.arange(1, len(sorted_q) + 1) / (len(sorted_q) + 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perc, y=sorted_q,
        mode="lines", line=dict(color="#2563eb", width=2),
        hovertemplate="Excedência: %{x:.1f}%<br>Q = %{y:.1f} m³/s<extra></extra>",
    ))
    for p, label in [(5, "Q5"), (50, "Q50"), (95, "Q95")]:
        v = np.percentile(q, 100 - p)
        fig.add_vline(
            x=p, line_dash="dot", line_color="#94a3b8",
            annotation_text=f"{label} = {v:.1f} m³/s",
            annotation_position="top right",
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Probabilidade de excedência (%)",
        yaxis_title="Q (m³/s)",
        yaxis_type="log",
        height=460,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Curva de permanência calculada sobre toda a série horária disponível "
        "(2021-2025). Eixo Y em escala logarítmica. Q5: vazão excedida em 5% "
        "do tempo (pico); Q50: mediana; Q95: vazão excedida em 95% do tempo "
        "(referência para disponibilidade hídrica em estudos de outorga)."
    )

# ======================================================= Sazonalidade
with tab_saz:
    serie["month"] = serie.index.month
    months = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
              "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    fig = go.Figure()
    for m in range(1, 13):
        fig.add_trace(go.Box(
            y=serie.loc[serie["month"] == m, "Q_obs"],
            name=months[m - 1],
            marker_color="#2563eb",
            boxmean=True,
        ))
    fig.update_layout(
        template="plotly_white",
        yaxis_title="Q (m³/s)",
        yaxis_type="log",
        height=440,
        showlegend=False,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Precipitação mensal média
    p_month = serie["P_mean"].resample("ME").sum()
    p_month_byyear = p_month.groupby(p_month.index.month).mean()
    fig_p = go.Figure(go.Bar(
        x=months, y=p_month_byyear.values,
        marker_color="#0ea5e9", opacity=0.85,
        text=[f"{v:.0f}" for v in p_month_byyear.values],
        textposition="outside",
    ))
    fig_p.update_layout(
        title="Precipitação mensal média (mm/mês)",
        template="plotly_white",
        yaxis_title="P (mm)",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_p, use_container_width=True)
    st.caption(
        "Distribuição mensal da vazão (boxplot em escala log) e precipitação "
        "acumulada média. Estação chuvosa concentrada entre outubro e março; "
        "estiagem marcada entre junho e setembro — padrão típico da região "
        "do Paraíba do Sul."
    )

# ==================================================== Disponibilidade
with tab_disp:
    df_disp = serie.copy()
    df_disp["year"] = df_disp.index.year
    df_disp["month"] = df_disp.index.month
    df_disp["valid"] = df_disp["Q_obs"].notna().astype(int)
    pivot = df_disp.pivot_table(
        values="valid", index="year", columns="month", aggfunc="mean"
    )
    pivot.columns = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                     "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#dc2626"], [0.5, "#fbbf24"], [1, "#16a34a"]],
        colorbar=dict(title="Disponibilidade", tickformat=".0%"),
        zmin=0, zmax=1,
        hovertemplate="%{y} · %{x}<br>Disponibilidade: %{z:.0%}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        height=max(320, len(pivot) * 36),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Disponibilidade da série horária de vazão por mês. Valores próximos "
        "de 100% indicam meses com dados completos. Lacunas resultaram em "
        "exclusão prévia de anos com disponibilidade inferior a 70%."
    )

# ========================================================= Ottobacias
with tab_otto:
    col1, col2 = st.columns(2)
    with col1:
        fig_cn = go.Figure(go.Histogram(
            x=attrs["cn_2022"], nbinsx=30,
            marker_color="#2563eb", opacity=0.85,
            hovertemplate="CN: %{x}<br>Ottobacias: %{y}<extra></extra>",
        ))
        fig_cn.add_vline(
            x=attrs["cn_2022"].mean(), line_dash="dash", line_color="#dc2626",
            annotation_text=f"Média = {attrs['cn_2022'].mean():.1f}",
            annotation_position="top right",
        )
        fig_cn.update_layout(
            title="Curve Number (BHAE_CN-2022)",
            template="plotly_white",
            xaxis_title="CN", yaxis_title="Número de ottobacias",
            height=360, margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_cn, use_container_width=True)

    with col2:
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Histogram(
            x=attrs["tc_base_h"], nbinsx=30,
            marker_color="#60a5fa", opacity=0.7,
            name="Tc base (Maidment)",
        ))
        fig_tc.add_trace(go.Histogram(
            x=attrs["tc_manning_h"], nbinsx=30,
            marker_color="#16a34a", opacity=0.7,
            name="Tc Manning (LULC)",
        ))
        fig_tc.update_layout(
            title="Tempo de concentração (horas)",
            template="plotly_white",
            xaxis_title="Tc (h)", yaxis_title="Número de ottobacias",
            height=360, barmode="overlay",
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_tc, use_container_width=True)

    # Scatter area × CN
    fig_sc = go.Figure(go.Scatter(
        x=attrs["area_km2"], y=attrs["cn_2022"],
        mode="markers",
        marker=dict(
            size=8,
            color=attrs["tc_manning_h"],
            colorscale="Viridis",
            colorbar=dict(title="Tc Manning<br>(horas)"),
            line=dict(color="white", width=0.5),
        ),
        hovertemplate=(
            "Ottobacia %{text}<br>Área: %{x:.1f} km²<br>"
            "CN: %{y:.0f}<br>Tc Manning: %{marker.color:.1f} h<extra></extra>"
        ),
        text=attrs["ottobacia_id"],
    ))
    fig_sc.update_layout(
        title="Atributos das 245 ottobacias: área × CN × Tc Manning",
        template="plotly_white",
        xaxis_title="Área (km²)",
        yaxis_title="Curve Number",
        height=440,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.caption(
        "Distribuição dos atributos fisiográficos das 245 ottobacias. "
        "CN mediano = 63 (predomínio de pastagens em solos bem desenvolvidos); "
        "Tc Manning mediano = 23,4 h (ajustado pela rugosidade da cobertura "
        "florestal na Serra da Mantiqueira). O Tc Manning é 2,5× maior que o "
        "Tc base topográfico por efeito da densa cobertura vegetal nas "
        "cabeceiras."
    )
