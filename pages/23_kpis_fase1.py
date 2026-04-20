"""
Widget: KPIs de cabecalho da Fase 1 — quatro cards com metricas agregadas.

Replica o bloco de KPIs da pagina /resultados do app principal:
- Melhor previsao 6 h (NSE e modelo)
- Melhor simulacao continua (NSE e modelo)
- Contagem de configuracoes Muito Bom (NSE 6 h >= 0,75)
- Ganho do melhor sobre o pior (pontos percentuais de NSE 6 h)

URL: /kpis_fase1
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="KPIs da Fase 1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data(show_spinner=False)
def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = pd.read_csv(DATA_DIR / "summary.csv")
    with open(DATA_DIR / "summary_continuous.json", encoding="utf-8") as f:
        cont_raw = json.load(f)
    continuous = pd.DataFrame([
        {
            "Modelo": m["model_name"],
            "NSE": m["metrics"]["nse"],
            "KGE": m["metrics"]["kge"],
        }
        for m in cont_raw
    ])
    return forecast, continuous


forecast, continuous = _load()

# Top / bottom
best_f = forecast.loc[forecast["NSE_6h"].idxmax()]
worst_f = forecast.loc[forecast["NSE_6h"].idxmin()]
best_c = continuous.loc[continuous["NSE"].idxmax()]

count_vb = int((forecast["NSE_6h"] >= 0.75).sum())
total = len(forecast)
gap_pp = float((best_f["NSE_6h"] - worst_f["NSE_6h"]) * 100)


def _label(nome: str) -> str:
    return nome.replace("LSTM_TTD_", "").replace("LSTM_", "").replace("_", " ")


st.markdown("##### Síntese da Fase 1 — quatro indicadores agregados")

# 4 cards em colunas
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;"
        f"border-left:5px solid #16a34a;padding:12px 14px;border-radius:4px;'>"
        f"<div style='font-size:11px;color:#166534;text-transform:uppercase;"
        f"letter-spacing:0.5px;'>Melhor — previsão 6 h</div>"
        f"<div style='font-size:26px;font-weight:700;color:#14532d;margin-top:4px;'>"
        f"NSE = {best_f['NSE_6h']:.3f}</div>"
        f"<div style='font-size:12px;color:#166534;margin-top:2px;'>"
        f"{_label(best_f['Modelo'])}</div>"
        f"<div style='font-size:11px;color:#166534;margin-top:2px;'>"
        f"Classificação: <b>Muito Bom</b></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"<div style='background:#eff6ff;border:1px solid #bfdbfe;"
        f"border-left:5px solid #2563eb;padding:12px 14px;border-radius:4px;'>"
        f"<div style='font-size:11px;color:#1e40af;text-transform:uppercase;"
        f"letter-spacing:0.5px;'>Melhor — simulação contínua</div>"
        f"<div style='font-size:26px;font-weight:700;color:#1e3a8a;margin-top:4px;'>"
        f"NSE = {best_c['NSE']:.3f}</div>"
        f"<div style='font-size:12px;color:#1e40af;margin-top:2px;'>"
        f"{_label(best_c['Modelo'])}</div>"
        f"<div style='font-size:11px;color:#1e40af;margin-top:2px;'>"
        f"KGE = {best_c['KGE']:.3f} · Muito Bom</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"<div style='background:#fefce8;border:1px solid #fde68a;"
        f"border-left:5px solid #ca8a04;padding:12px 14px;border-radius:4px;'>"
        f"<div style='font-size:11px;color:#854d0e;text-transform:uppercase;"
        f"letter-spacing:0.5px;'>Configurações <i>Muito Bom</i></div>"
        f"<div style='font-size:26px;font-weight:700;color:#713f12;margin-top:4px;'>"
        f"{count_vb} / {total}</div>"
        f"<div style='font-size:12px;color:#854d0e;margin-top:2px;'>"
        f"NSE 6 h ≥ 0,75 (Moriasi)</div>"
        f"<div style='font-size:11px;color:#854d0e;margin-top:2px;'>"
        f"todas distribuídas com TTD</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"<div style='background:#fdf2f8;border:1px solid #fbcfe8;"
        f"border-left:5px solid #db2777;padding:12px 14px;border-radius:4px;'>"
        f"<div style='font-size:11px;color:#9d174d;text-transform:uppercase;"
        f"letter-spacing:0.5px;'>Ganho vs pior configuração</div>"
        f"<div style='font-size:26px;font-weight:700;color:#831843;margin-top:4px;'>"
        f"+{gap_pp:.0f} pp</div>"
        f"<div style='font-size:12px;color:#9d174d;margin-top:2px;'>"
        f"NSE 6 h</div>"
        f"<div style='font-size:11px;color:#9d174d;margin-top:2px;'>"
        f"{_label(best_f['Modelo'])} vs {_label(worst_f['Modelo'])}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("")

st.caption(
    "As 10 configurações do estudo comparativo foram treinadas por 300 épocas "
    "(paciência 30) sobre o período 2021–2024 e avaliadas em 2025. A bacia de "
    "3.117 km² foi discretizada em 245 ottobacias. A classificação de NSE "
    "segue Moriasi *et al.* (2007): Insatisfatório (&lt; 0,50), Satisfatório "
    "(0,50–0,65), Bom (0,65–0,75) e Muito Bom (≥ 0,75)."
)
