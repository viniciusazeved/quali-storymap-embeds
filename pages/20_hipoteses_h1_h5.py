"""
Widget: veredito das 5 hipoteses operacionais (H1-H5) da Fase 1.

Replica a aba "Hipoteses H1-H5" da pagina /resultados do app principal.
Cada hipotese tem veredito DUPLO (previsao 6 h e simulacao continua)
com icone, rotulo e evidencia numerica.

Destaque: o trade-off ajustavel x fixo nao estava entre as hipoteses
originais — e achado nao previsto da Fase 1 e motiva a Fase 2.

URL: /hipoteses_h1_h5
"""
from __future__ import annotations

import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Hipóteses H1–H5 — TTD-SCS-LSTM",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()


HYPOTHESES = [
    {
        "id": "H1",
        "name": "Distribuído &gt; Concentrado",
        "test": "LSTM vs LSTM_Lumped",
        "prev": ("✅", "Confirmada", "NSE 6 h: 0,62 vs 0,54"),
        "cont": ("✅", "Confirmada", "NSE contínuo: 0,67 vs 0,61"),
        "summary": (
            "A configuração distribuída (245 ottobacias) supera a concentrada "
            "em <b>ambas as abordagens</b> — a discretização espacial preserva "
            "a heterogeneidade da bacia e se traduz em ganho de NSE em "
            "previsão e simulação contínua."
        ),
    },
    {
        "id": "H2",
        "name": "Manning &gt; Base para <i>T<sub>c</sub></i>",
        "test": "LSTM_TTD_Manning vs LSTM_TTD_Base",
        "prev": ("❌", "Rejeitada", "NSE 6 h: Manning 0,83 vs Base 0,84"),
        "cont": ("✅", "Confirmada", "NSE contínuo: Manning 0,81 vs Base 0,70"),
        "summary": (
            "Na previsão de 6 h, Base supera Manning por margem mínima (0,01). "
            "Na <b>simulação contínua, Manning supera significativamente</b> "
            "(+15,7 %), evidenciando que a rugosidade por uso do solo confere "
            "robustez para aplicações de longo prazo."
        ),
    },
    {
        "id": "H3",
        "name": "Parâmetros ajustáveis &gt; fixos",
        "test": "Pares aprendível vs fixo",
        "prev": ("✅", "Confirmada", "média: +18,3 % NSE 6 h"),
        "cont": ("❌", "Rejeitada", "para TTD Base: fixo 0,82 &gt; ajust 0,70"),
        "summary": (
            "Na previsão, parâmetros ajustáveis melhoram o NSE em 18,3 % na "
            "média. Na simulação contínua, o efeito depende da formulação de "
            "<i>T<sub>c</sub></i>: para TTD Base, parâmetros fixos superam os "
            "ajustáveis (0,82 vs 0,70); para TTD Manning, parâmetros ajustáveis "
            "são superiores (0,81 vs 0,54); para TTD Base+SCS, ajustáveis "
            "também superam (0,76 vs 0,67). Este comportamento assimétrico "
            "caracteriza o <i>trade-off</i> entre otimização de curto prazo "
            "e generalização temporal, discutido na Seção 4.7 da qualificação."
        ),
    },
    {
        "id": "H4",
        "name": "SCS-CN agrega valor",
        "test": "Com SCS vs sem SCS (em cada combinação)",
        "prev": ("❌", "Rejeitada", "SCS degrada 6–9,5 % em 6 h"),
        "cont": ("⚠️", "Parcial", "só agrega em TTD_Base ajustável (+8,6 %)"),
        "summary": (
            "Em bacia única, o SCS-CN é <b>parcialmente redundante</b>: a "
            "LSTM com 240 h de histórico já captura implicitamente a "
            "transformação chuva-escoamento. O CN permanece na arquitetura "
            "pois será <b>crítico na Fase 2</b> — é o atributo que "
            "diferencia bacias com solos e uso do solo distintos."
        ),
    },
    {
        "id": "H5",
        "name": "Modelo completo &gt; modelos de referência",
        "test": "TTD-SCS-LSTM vs LSTM puro / Lumped",
        "prev": ("✅", "Confirmada", "+35,5 % vs LSTM; +55,6 % vs Lumped"),
        "cont": ("✅", "Confirmada", "+21,0 % vs LSTM; +34,4 % vs Lumped"),
        "summary": (
            "O modelo híbrido supera os modelos de referência em <b>ambas as "
            "abordagens</b> e em margem expressiva — demonstra que a física "
            "codificada agrega informação mesmo em bacia única. Expectativa: "
            "ganho ainda maior em aplicação multi-bacia (Fase 2)."
        ),
    },
]

st.markdown(
    "##### Veredito das 5 hipóteses operacionais — estudo comparativo da Fase 1"
)

st.caption(
    "Cada hipótese foi formulada no Capítulo 1 e testada sistematicamente pelo "
    "estudo comparativo (ablação). Veredito duplo: previsão 6 h e simulação contínua."
)

for h in HYPOTHESES:
    st.markdown(
        f"<div style='background:#f8fafc;border-left:4px solid #0ea5e9;"
        f"padding:10px 14px;border-radius:4px;margin-top:10px;'>"
        f"<div style='font-weight:700;font-size:15px;color:#0f172a;'>"
        f"<span style='background:#0ea5e9;color:white;padding:2px 8px;"
        f"border-radius:3px;margin-right:6px;'>{h['id']}</span>"
        f"{h['name']}"
        f"</div>"
        f"<div style='font-size:11px;color:#64748b;margin-top:2px;'>"
        f"Teste: <i>{h['test']}</i></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    icon_p, verd_p, ev_p = h["prev"]
    icon_c, verd_c, ev_c = h["cont"]
    with c1:
        st.markdown(
            f"**Previsão 6 h:** {icon_p} {verd_p}  \n"
            f"<span style='font-size:12px;color:#64748b;'>{ev_p}</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"**Simulação contínua:** {icon_c} {verd_c}  \n"
            f"<span style='font-size:12px;color:#64748b;'>{ev_c}</span>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<div style='font-size:13px;color:#334155;margin-top:4px;'>"
        f"{h['summary']}</div>",
        unsafe_allow_html=True,
    )

st.divider()

st.markdown(
    "<div style='background:#fef3c7;border-left:4px solid #ca8a04;"
    "padding:10px 14px;border-radius:4px;font-size:13px;color:#713f12;'>"
    "<b>Achado não previsto — trade-off previsão × contínua.</b> A caracterização "
    "numérica do trade-off entre parâmetros aprendíveis (melhor previsão) e "
    "parâmetros fixos (melhor simulação contínua) não constava das hipóteses "
    "originais. É o <b>resultado mais expressivo</b> da Fase 1 e motiva "
    "diretamente a Fase 2 (regionalização multi-bacia), onde a generalização "
    "espacial será avaliada em ~100 bacias."
    "</div>",
    unsafe_allow_html=True,
)
