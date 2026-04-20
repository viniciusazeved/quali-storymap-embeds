"""
Widget: eventos de cheia com comparacao melhor × pior (forecast 6 h).

Variante do /hidrograma_eventos focada em contraste entre a melhor e a
pior configuracao da previsao 6 horas:
- Melhor: LSTM_TTD_Base (NSE = 0.837, distribuido + TTD aprendivel)
- Pior: LSTM_Lumped (NSE = 0.542, concentrado sem discretizacao espacial)

Detecta automaticamente picos acima do Q95 com separacao de 48 h,
permite selecionar um evento e exibe as duas predicoes sobrepostas.

URL: /hidrograma_eventos_
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
    page_title="Eventos — melhor × pior",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_FILE = Path(__file__).parent.parent / "data" / "hidrogramas_v2.csv"

# Configuracoes em ordem de desempenho (melhor -> pior)
MODELOS = [
    {
        "coluna": "Q_pred_Base_6h",
        "label": "Melhor — Base",
        "descricao": "LSTM + TTD Base aprendível",
        "cor": "#16a34a",
    },
    {
        "coluna": "Q_pred_Lumped_6h",
        "label": "Pior — Lumped",
        "descricao": "LSTM concentrado, sem discretização espacial",
        "cor": "#dc2626",
    },
]


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    return df.set_index("timestamp").sort_index()


df = _load()

# Blindagem — evita KeyError quando o Cloud esta com deploy parcial (codigo
# novo mas CSV antigo). Mostra mensagem acionavel em vez de crashar.
_REQUIRED = ["Q_obs_6h", "P_mean"] + [m["coluna"] for m in MODELOS]
_missing = [c for c in _REQUIRED if c not in df.columns]
if _missing:
    st.error(
        f"CSV `hidrogramas.csv` sem as colunas esperadas: "
        f"{', '.join(_missing)}. Se você acabou de fazer deploy, faça "
        "*Reboot* no painel do Streamlit Cloud para recarregar os dados."
    )
    st.stop()

show_precip = st.checkbox("Mostrar precipitação", value=True)

obs = df["Q_obs_6h"]
precip = df["P_mean"]

# Q95 e deteccao de picos
q95 = float(np.nanpercentile(obs.dropna().values, 95))
obs_arr = obs.values.copy()
obs_arr_filled = np.where(np.isnan(obs_arr), -np.inf, obs_arr)

peaks: list[int] = []
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
    f"**Q95 = {q95:.1f} m³/s** · **{len(peaks)} evento(s)** detectado(s) "
    f"com pico acima do Q95 (separação mínima de 48 h)."
)

if not peaks:
    st.info("Nenhum evento de cheia detectado.")
    st.stop()

event_options = [
    f"Evento {k + 1}: {obs.index[p]:%d/%m/%Y %H:%M}  ·  Q_pico = {obs_arr[p]:.1f} m³/s"
    for k, p in enumerate(peaks)
]
evt = st.selectbox("Selecione um evento", event_options, index=0)
idx = event_options.index(evt)
peak_pos = peaks[idx]

# Zoom 72h antes, 120h depois
i0 = max(0, peak_pos - 72)
i1 = min(len(obs), peak_pos + 120)
obs_e = obs.iloc[i0:i1]
precip_e = precip.iloc[i0:i1]

# KPIs por modelo
def _metricas_evento(o_full: pd.Series, p_full: pd.Series) -> dict[str, float]:
    o_valid = o_full.dropna()
    o_ = o_valid.values
    p_ = p_full.reindex(o_valid.index).values
    valid = ~np.isnan(p_)
    o_ = o_[valid]
    p_ = p_[valid]
    if len(o_) < 10:
        return {k: np.nan for k in ("nse", "pbias", "pico_o", "pico_p", "err_p", "err_vol")}
    nse = 1 - np.sum((o_ - p_) ** 2) / np.sum((o_ - o_.mean()) ** 2)
    pbias = 100 * (p_.sum() - o_.sum()) / o_.sum()
    pico_o = float(o_.max())
    pico_p = float(p_.max())
    err_p = 100 * (pico_p - pico_o) / pico_o if pico_o > 0 else np.nan
    vol_o = float(np.trapezoid(o_))
    vol_p = float(np.trapezoid(p_))
    err_vol = 100 * (vol_p - vol_o) / vol_o if vol_o > 0 else np.nan
    return {
        "nse": float(nse), "pbias": float(pbias),
        "pico_o": pico_o, "pico_p": pico_p,
        "err_p": float(err_p), "err_vol": float(err_vol),
    }


kpi_cols = st.columns(len(MODELOS))
for col, cfg in zip(kpi_cols, MODELOS):
    pred_full = df[cfg["coluna"]].iloc[i0:i1]
    m = _metricas_evento(obs_e, pred_full)
    with col:
        st.markdown(
            f"<div style='border-left:4px solid {cfg['cor']};padding:6px 10px;"
            f"background:#f8fafc;border-radius:4px;'>"
            f"<div style='font-weight:600;color:#0f172a;font-size:13px;'>{cfg['label']}</div>"
            f"<div style='font-size:11px;color:#64748b;margin-bottom:6px;'>{cfg['descricao']}</div>"
            f"<div style='font-size:12px;line-height:1.5;'>"
            f"NSE: <b>{m['nse']:.3f}</b> &nbsp;·&nbsp; "
            f"PBIAS: {m['pbias']:+.1f}%<br>"
            f"Pico pred: {m['pico_p']:.1f} m³/s "
            f"(erro {m['err_p']:+.1f}%)<br>"
            f"Erro volume: {m['err_vol']:+.1f}%"
            f"</div></div>",
            unsafe_allow_html=True,
        )

# Linha com o pico observado (comum aos dois)
st.markdown(
    f"<div style='font-size:12px;color:#64748b;margin-top:4px;'>"
    f"Pico observado: <b style='color:#0f172a;'>"
    f"{float(obs_e.dropna().max()):.1f} m³/s</b>"
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# Hidrograma do evento
if show_precip:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.22, 0.78], vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Bar(
            x=precip_e.index, y=precip_e.values,
            marker_color="#0ea5e9", opacity=0.7,
            name="Precipitação",
            hovertemplate="%{x|%d/%m %H:%M}<br>P = %{y:.2f} mm/h<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title_text="P (mm/h)", autorange="reversed",
                     row=1, col=1, gridcolor="#f1f5f9")
    row_q = 2
else:
    fig = go.Figure()
    row_q = None

# Observado em preto
t_obs = go.Scatter(
    x=obs_e.index, y=obs_e.values,
    mode="lines", line=dict(color="#0f172a", width=2.4),
    name="Observado",
    hovertemplate="%{x|%d/%m %H:%M}<br>Q_obs = %{y:.1f} m³/s<extra></extra>",
)
if row_q is not None:
    fig.add_trace(t_obs, row=row_q, col=1)
else:
    fig.add_trace(t_obs)

# Duas predicoes sobrepostas
for cfg in MODELOS:
    pred_full = df[cfg["coluna"]].iloc[i0:i1]
    tr = go.Scatter(
        x=pred_full.index, y=pred_full.values,
        mode="lines", line=dict(color=cfg["cor"], width=2.0),
        name=cfg["label"],
        hovertemplate=(
            f"<b>{cfg['label']}</b><br>"
            "%{x|%d/%m %H:%M}<br>"
            "Q_pred = %{y:.1f} m³/s"
            "<extra></extra>"
        ),
    )
    if row_q is not None:
        fig.add_trace(tr, row=row_q, col=1)
    else:
        fig.add_trace(tr)

# Linha vertical no pico
peak_time = obs.index[peak_pos]
if row_q is not None:
    fig.add_vline(x=peak_time, line_dash="dot", line_color="#64748b",
                  opacity=0.6, row=row_q, col=1)
    fig.update_yaxes(title_text="Q (m³/s)", row=row_q, col=1, gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", row=row_q, col=1, gridcolor="#f1f5f9")
else:
    fig.add_vline(x=peak_time, line_dash="dot", line_color="#64748b", opacity=0.6)
    fig.update_yaxes(title_text="Q (m³/s)", gridcolor="#f1f5f9")
    fig.update_xaxes(title_text="Tempo", gridcolor="#f1f5f9")

fig.update_layout(
    template="plotly_white",
    title=f"Evento em {peak_time:%d de %B de %Y, %H:%M}",
    height=470,
    hovermode="x unified",
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01,
        xanchor="right", x=1, bgcolor="rgba(255,255,255,0.7)",
    ),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Previsão 6 horas à frente, janela de 72 h antes e 120 h depois do pico "
    "observado. A configuração `Base` (melhor, NSE geral = 0,837) acompanha "
    "de perto a forma e o instante do pico observado; a configuração `Lumped` "
    "(pior, NSE geral = 0,542) apresenta subestimativa consistente da "
    "amplitude e defasagem temporal perceptível, reflexo da ausência de "
    "discretização espacial e do módulo de propagação temporal TTD."
)
