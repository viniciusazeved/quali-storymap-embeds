"""
Widget: explorador interativo de hidrogramas — observado x predito, com
deteccao automatica de eventos de cheia.

Funcionalidades:
  - Seletor de configuracao (Base, Base_Fixed, Manning) e horizonte (1h, 6h, 24h)
  - Janela temporal ajustavel via slider
  - Hidrograma principal (obs vs pred + precipitacao opcional)
  - KPIs da janela (NSE, RMSE, pico obs, pico pred, erro do pico)
  - Scatter observado x predito (com linha 1:1)
  - Serie de residuos temporais
  - Deteccao automatica de eventos de cheia (pico > Q95 com separacao 48 h)
  - Selecao de evento com zoom (72 h antes, 120 h depois) + KPIs do evento

URL: /explorador_hidrogramas
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
    page_title="Explorador de hidrogramas",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_FILE = Path(__file__).parent.parent / "data" / "hidrogramas.csv"

MODEL_OPTIONS = {
    "LSTM + TTD Base (aprendível)": "Base",
    "LSTM + TTD Base Fixed (parâmetros fixos)": "Base_Fixed",
    "LSTM + TTD Manning (aprendível)": "Manning",
}
HORIZON_OPTIONS = ["1h", "6h", "24h"]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    return df.set_index("timestamp").sort_index()


df = _load()
ts_min = df.index.min().to_pydatetime()
ts_max = df.index.max().to_pydatetime()

# ---------------------------------------------------------------------------
# Controles no topo
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1.5, 1])
with c1:
    modelo_label = st.selectbox(
        "Configuração",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )
    modelo_key = MODEL_OPTIONS[modelo_label]
with c2:
    horizonte = st.radio(
        "Horizonte de previsão",
        HORIZON_OPTIONS,
        index=1,
        horizontal=True,
    )
with c3:
    show_precip = st.checkbox("Precipitação", value=True)

# Slider de janela temporal
default_start = pd.Timestamp("2025-09-01").to_pydatetime()
default_start = max(default_start, ts_min)
janela = st.slider(
    "Janela temporal",
    min_value=ts_min,
    max_value=ts_max,
    value=(default_start, ts_max),
    format="YYYY-MM-DD",
)

# ---------------------------------------------------------------------------
# Filtra janela
# ---------------------------------------------------------------------------
mask = (df.index >= janela[0]) & (df.index <= janela[1])
dfw = df.loc[mask].copy()

obs_col = f"Q_obs_{horizonte}"
pred_col = f"Q_pred_{modelo_key}_{horizonte}"
obs = dfw[obs_col]
pred = dfw[pred_col]
precip = dfw["P_mean"]


# ---------------------------------------------------------------------------
# KPIs da janela (antes do grafico)
# ---------------------------------------------------------------------------
valid = (~obs.isna()) & (~pred.isna())
o = obs[valid].values
p = pred[valid].values

if len(o) > 10:
    nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((o - p) ** 2)))
    pbias = float(100 * (p.sum() - o.sum()) / o.sum())
    r2 = float(np.corrcoef(o, p)[0, 1]) ** 2
    pico_obs = float(o.max())
    pico_pred = float(p.max())
    err_pico = 100.0 * (pico_pred - pico_obs) / pico_obs if pico_obs > 0 else float("nan")
else:
    nse = rmse = pbias = r2 = pico_obs = pico_pred = err_pico = float("nan")

k1, k2, k3, k4 = st.columns(4)
k1.metric("NSE", f"{nse:.3f}" if np.isfinite(nse) else "—")
k2.metric("RMSE (m³/s)", f"{rmse:.2f}" if np.isfinite(rmse) else "—")
k3.metric("PBIAS (%)", f"{pbias:+.2f}" if np.isfinite(pbias) else "—")
k4.metric("R²", f"{r2:.3f}" if np.isfinite(r2) else "—")

st.divider()

# ---------------------------------------------------------------------------
# Hidrograma principal
# ---------------------------------------------------------------------------
def _build_hydrograph(obs_s, pred_s, precip_s, title, height=440):
    """Hidrograma obs vs pred com precipitação opcional no topo invertida."""
    if precip_s is not None:
        f = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.22, 0.78],
            vertical_spacing=0.03,
        )
        f.add_trace(
            go.Bar(
                x=precip_s.index, y=precip_s.values,
                marker_color="#0ea5e9", opacity=0.7,
                name="Precipitação",
                hovertemplate="%{x|%d/%m %H:%M}<br>P = %{y:.2f} mm/h<extra></extra>",
            ),
            row=1, col=1,
        )
        f.update_yaxes(
            title_text="P (mm/h)", autorange="reversed",
            row=1, col=1, gridcolor="#f1f5f9",
        )
        row_q = 2
    else:
        f = go.Figure()
        row_q = None

    t_obs = go.Scatter(
        x=obs_s.index, y=obs_s.values,
        mode="lines", line=dict(color="#0f172a", width=2),
        name="Observado",
        hovertemplate="%{x|%d/%m %H:%M}<br>Q_obs = %{y:.1f} m³/s<extra></extra>",
    )
    t_pred = go.Scatter(
        x=pred_s.index, y=pred_s.values,
        mode="lines", line=dict(color="#2563eb", width=2),
        name="Predito",
        hovertemplate="%{x|%d/%m %H:%M}<br>Q_pred = %{y:.1f} m³/s<extra></extra>",
    )
    if row_q is not None:
        f.add_trace(t_obs, row=row_q, col=1)
        f.add_trace(t_pred, row=row_q, col=1)
        f.update_yaxes(title_text="Q (m³/s)", row=row_q, col=1, gridcolor="#f1f5f9")
        f.update_xaxes(title_text="Tempo", row=row_q, col=1, gridcolor="#f1f5f9")
    else:
        f.add_trace(t_obs)
        f.add_trace(t_pred)
        f.update_yaxes(title_text="Q (m³/s)", gridcolor="#f1f5f9")
        f.update_xaxes(title_text="Tempo", gridcolor="#f1f5f9")

    f.update_layout(
        title=title if title else None,
        template="plotly_white",
        height=height,
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)",
        ),
        plot_bgcolor="white",
    )
    return f


precip_plot = precip if show_precip else None
fig = _build_hydrograph(obs, pred, precip_plot, title=None, height=460)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tabs analíticas
# ---------------------------------------------------------------------------
tab_scatter, tab_eventos = st.tabs([
    "Scatter e resíduos",
    "Eventos de cheia",
])

# ==================================================== Scatter + Resíduos
with tab_scatter:
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Observado × Predito**")
        if len(o) > 0:
            lim = max(o.max(), p.max()) * 1.05
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=o, y=p,
                mode="markers",
                marker=dict(color="#2563eb", size=5, opacity=0.45,
                            line=dict(color="white", width=0.3)),
                name="Pares",
                hovertemplate="Obs: %{x:.1f}<br>Pred: %{y:.1f}<extra></extra>",
            ))
            fig_sc.add_trace(go.Scatter(
                x=[0, lim], y=[0, lim],
                mode="lines", line=dict(color="#dc2626", dash="dash", width=2),
                name="Identidade (1:1)",
            ))
            fig_sc.update_layout(
                template="plotly_white",
                xaxis=dict(title="Q observado (m³/s)", range=[0, lim], gridcolor="#f1f5f9"),
                yaxis=dict(title="Q predito (m³/s)", range=[0, lim], gridcolor="#f1f5f9"),
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    with col_s2:
        st.markdown("**Resíduos (predito − observado)**")
        res = (pred - obs).dropna()
        if len(res) > 0:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=res.index, y=res.values,
                mode="lines", line=dict(color="#dc2626", width=1),
                hovertemplate="%{x|%d/%m %H:%M}<br>Resíduo: %{y:.1f} m³/s<extra></extra>",
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#475569")
            fig_res.update_layout(
                template="plotly_white",
                xaxis=dict(title="Tempo", gridcolor="#f1f5f9"),
                yaxis=dict(title="Resíduo (m³/s)", gridcolor="#f1f5f9"),
                height=400,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_res, use_container_width=True)

    st.caption(
        "**Scatter:** pontos próximos à linha 1:1 indicam boa concordância entre "
        "o observado e o predito. Dispersão sistemática acima ou abaixo da linha "
        "revela viés. **Resíduos:** padrões temporais nos resíduos (picos, "
        "viés sustentado) apontam para limitações específicas do modelo na janela."
    )

# ====================================================== Eventos de cheia
with tab_eventos:
    # Q95 da janela
    q95 = float(np.nanpercentile(obs.dropna().values, 95)) if len(obs.dropna()) > 0 else float("nan")

    # Deteccao de picos: obs > q95, máximo local, separação de 48h
    obs_arr = obs.values.copy()
    obs_arr_filled = np.where(np.isnan(obs_arr), -np.inf, obs_arr)
    peaks = []
    i = 1
    while i < len(obs_arr_filled) - 1:
        if (obs_arr_filled[i] > q95
                and obs_arr_filled[i] > obs_arr_filled[i - 1]
                and obs_arr_filled[i] > obs_arr_filled[i + 1]):
            peaks.append(i)
            i += 48
        else:
            i += 1

    st.markdown(
        f"**Q95 da janela = {q95:.1f} m³/s** · "
        f"**{len(peaks)} evento(s)** com pico acima do Q95 detectado(s)."
    )

    if peaks:
        event_options = [
            f"Evento {k + 1}: {obs.index[p]:%Y-%m-%d %H:%M} "
            f"(Q_pico = {obs_arr[p]:.1f} m³/s)"
            for k, p in enumerate(peaks)
        ]
        evt = st.selectbox("Selecione um evento", event_options, index=0)
        idx = event_options.index(evt)
        peak_pos = peaks[idx]

        # Janela: 72h antes, 120h depois
        i0 = max(0, peak_pos - 72)
        i1 = min(len(obs), peak_pos + 120)
        obs_e = obs.iloc[i0:i1]
        pred_e = pred.iloc[i0:i1]
        precip_e = precip.iloc[i0:i1] if show_precip else None

        fig_e = _build_hydrograph(
            obs_e, pred_e, precip_e,
            title=f"Evento em {obs.index[peak_pos]:%d de %B de %Y, %H:%M}",
            height=420,
        )
        st.plotly_chart(fig_e, use_container_width=True)

        # KPIs do evento
        o_e = obs_e.dropna().values
        p_e = pred_e.reindex(obs_e.dropna().index).values
        valid_e = ~np.isnan(p_e)
        o_e = o_e[valid_e]
        p_e = p_e[valid_e]

        if len(o_e) > 10:
            nse_e = 1 - np.sum((o_e - p_e) ** 2) / np.sum((o_e - o_e.mean()) ** 2)
            pbias_e = 100 * (p_e.sum() - o_e.sum()) / o_e.sum()
            pico_o = float(o_e.max())
            pico_p = float(p_e.max())
            err_p = 100 * (pico_p - pico_o) / pico_o if pico_o > 0 else float("nan")
            vol_o = float(np.trapz(o_e))
            vol_p = float(np.trapz(p_e))
            err_vol = 100 * (vol_p - vol_o) / vol_o if vol_o > 0 else float("nan")
        else:
            nse_e = pbias_e = pico_o = pico_p = err_p = err_vol = float("nan")

        ce1, ce2, ce3, ce4, ce5 = st.columns(5)
        ce1.metric("NSE (evento)", f"{nse_e:.3f}" if np.isfinite(nse_e) else "—")
        ce2.metric("PBIAS (%)", f"{pbias_e:+.2f}" if np.isfinite(pbias_e) else "—")
        ce3.metric("Pico observado (m³/s)", f"{pico_o:.1f}" if np.isfinite(pico_o) else "—")
        ce4.metric("Pico predito (m³/s)", f"{pico_p:.1f}" if np.isfinite(pico_p) else "—")
        ce5.metric("Erro do pico (%)", f"{err_p:+.1f}" if np.isfinite(err_p) else "—")

        st.caption(
            "Cada evento é isolado automaticamente a partir de máximos locais "
            "acima do Q95 da janela, com separação mínima de 48 h entre picos. "
            "A visualização mostra 72 h antes e 120 h depois do pico observado. "
            "Erro do pico: diferença percentual entre pico predito e pico observado "
            "— valores positivos indicam superestimativa."
        )
    else:
        st.info(
            "Nenhum evento de cheia detectado na janela atual. Amplie o intervalo "
            "temporal ou selecione uma janela que inclua a estação chuvosa "
            "(outubro a março)."
        )

st.caption(
    f"**Configuração:** {modelo_label} · **Horizonte:** {horizonte} à frente · "
    f"**Amostras válidas na janela:** {len(o):,} h."
    .replace(",", ".")
)
