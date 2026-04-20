"""
Landing page do app de embeds — serve como indice dos widgets disponiveis.

Os widgets sao consumidos via iframe pelo StoryMap ArcGIS do projeto
(https://storymaps.arcgis.com/stories/d3b585511b40427eb5fd1049a20b99c1).

Esta pagina e raramente acessada diretamente; e o destino-raiz do dominio
do Streamlit Cloud. As paginas individuais (em pages/) sao as URLs que
realmente entram no iframe do StoryMap.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="TTD-SCS-LSTM — Embeds",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Widgets do StoryMap — TTD-SCS-LSTM")

st.markdown(
    """
    Este app hospeda **widgets minimalistas** embedados no
    [StoryMap da qualificação](https://storymaps.arcgis.com/stories/d3b585511b40427eb5fd1049a20b99c1)
    via iframe. Cada página é uma visualização autônoma — sem narrativa,
    sem sidebar — projetada para viver dentro de blocos *Embed* ou
    *Sidecar* do StoryMap ArcGIS.

    O app principal da qualificação continua em
    [quali-ttd-scs-lstm.streamlit.app](https://quali-ttd-scs-lstm.streamlit.app).

    ### Widgets disponíveis

    | Widget | URL relativa | Uso no StoryMap |
    |---|---|---|
    | 🗺️ Mapa das estações telemétricas | `/mapa_telemetricas` | Seção "A rede telemétrica" |
    | 📊 Trade-off previsão × simulação | `/tradeoff` | Seção "O trade-off" |
    | 🎯 Melhor × pior por modo | `/melhor_pior_modo` | Seção de resultados |
    | 🎯 Hidrograma contínuo — melhor × pior | `/hidrograma_continuo_` | Seção de resultados |
    | 🎯 Eventos de cheia — melhor × pior | `/hidrograma_eventos_` | Seção de resultados |
    | 📊 Ranking NSE com slider de horizonte | `/ranking_nse` | Seção de resultados |

    Novos widgets vão sendo adicionados conforme a narrativa do StoryMap se
    desenha. Todos usam os mesmos dados do app principal (snapshot dos
    shapefiles ANA e CSV de resultados da Fase 1).
    """
)

st.info(
    "💡 Para embedar um widget, copie a URL completa "
    "(`https://<domain>.streamlit.app/mapa_telemetricas`, por exemplo) e "
    "cole em um bloco *Embed* do StoryMap ArcGIS."
)

st.caption(
    "Vinícius Azevedo · Qualificação de Doutorado · UNICAMP/FECFAU/DRH · "
    "defesa 22 de abril de 2026"
)
