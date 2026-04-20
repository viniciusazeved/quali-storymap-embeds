"""
Widget: limitacoes reconhecidas da Fase 1.

Replica a aba "Limitacoes" da pagina /resultados do app principal.
Cada limitacao em um expander — contextualiza os resultados e antecipa
criticas da banca, com a solucao ja mapeada.

URL: /limitacoes_fase1
"""
from __future__ import annotations

import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Limitações — Fase 1",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()


LIMITACOES = [
    {
        "titulo": "1. Validação em uma única bacia",
        "texto": (
            "Os resultados se referem à bacia de Manuel Duarte (3.117 km², "
            "245 ottobacias). **A generalização será avaliada na Fase 2** "
            "com ~100 bacias brasileiras — validação *leave-one-basin-out* "
            "e k-fold espacial."
        ),
    },
    {
        "titulo": "2. Período de dados de 5 anos",
        "texto": (
            "O período 2021–2025 pode não capturar toda a variabilidade "
            "climática interanual, especialmente ciclos plurianuais de seca. "
            "O conjunto de teste (abr–dez/2025) cobre predominantemente o "
            "período seco e a transição para o período chuvoso, sem incluir "
            "o pico da estação úmida (jan–mar). Na Fase 2, com "
            "aproximadamente 100 bacias e períodos mais longos, esta "
            "limitação será atenuada pela diversidade de condições hidrológicas."
        ),
    },
    {
        "titulo": "3. Escoamento de base implícito",
        "texto": (
            "Os módulos TTD e SCS-CN representam exclusivamente o escoamento "
            "direto. A componente lenta (*baseflow*) é capturada implicitamente "
            "pela memória de 240 h da LSTM. A separação explícita via filtro "
            "digital (Nathan, 1990; Eckhardt, 2005) é aprimoramento planejado."
        ),
    },
    {
        "titulo": "4. IUH gaussiano (simétrico)",
        "texto": (
            "O hidrograma unitário derivado do método de Maidment é tipicamente "
            "assimétrico (ascensão rápida, recessão lenta), enquanto o "
            "*kernel* gaussiano é simétrico. A escolha prioriza parcimônia "
            "e estabilidade numérica; a LSTM atua como compensador residual. "
            "Alternativas igualmente diferenciáveis, como distribuição gamma "
            "(Nash, 1957) ou log-normal, estão planejadas para investigação na Fase 2."
        ),
    },
    {
        "titulo": "5. Ausência de comparação com modelos externos",
        "texto": (
            "A Fase 1 comparou apenas configurações internas do *framework*, "
            "o que permite isolar a contribuição de cada componente. "
            "Comparação com GR4J, HBV e EA-LSTM (Kratzert, 2019) está "
            "planejada para o Artigo 1, essencial para contextualizar o "
            "desempenho no estado da arte internacional."
        ),
    },
    {
        "titulo": "6. Módulo hidrológico isolado não avaliado",
        "texto": (
            "O desempenho de $Q_{physics}$ (saída do TTD antes da LSTM) "
            "**não foi avaliado isoladamente**. Isso impede quantificar "
            "a contribuição corretiva da LSTM. Análise dos resíduos "
            "$(Q_{pred} - Q_{physics})$ está planejada para o Artigo 1."
        ),
    },
]


st.markdown("##### Limitações reconhecidas — estudo da Fase 1")

st.caption(
    "Extraídas da Seção 4.8 da qualificação. Contextualizam adequadamente "
    "os resultados e antecipam críticas da banca, com soluções já mapeadas "
    "para o Artigo 1 e a Fase 2."
)

for lim in LIMITACOES:
    with st.expander(f"**{lim['titulo']}**"):
        st.markdown(lim["texto"])

st.divider()

st.markdown(
    "<div style='background:#e0f2fe;border-left:4px solid #0ea5e9;"
    "padding:10px 14px;border-radius:4px;font-size:13px;color:#0c4a6e;'>"
    "<b>💡 Por que expor as limitações?</b> Ter cada limitação articulada "
    "— e <b>as soluções já mapeadas</b> — é sinal de maturidade metodológica. "
    "As Fases 2 e 3 e o Artigo 1 endereçam diretamente cada um destes seis "
    "pontos."
    "</div>",
    unsafe_allow_html=True,
)
