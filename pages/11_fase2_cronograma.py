"""
Widget: Fase 2 — regionalizacao multi-bacia e cronograma do doutorado.

Sintese do Capitulo 5 da qualificacao (cronograma.tex): motivacao da
extensao multi-bacia, etapas metodologicas e cronograma ate a defesa.
Replica a pagina "Fase 2" do app principal em tom academico, compativel
com o consumo via StoryMap ArcGIS.

URL: /fase2_cronograma
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Fase 2 — regionalização multi-bacia",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

st.markdown(
    """
    Síntese do Capítulo 5 da qualificação. A Fase 1 consolidou o
    modelo TTD-SCS-LSTM em bacia única (NSE = 0,84 em previsão de 6 h
    e NSE = 0,82 em simulação contínua no rio Preto). A Fase 2 propõe
    a extensão para regionalização em aproximadamente 100 bacias
    brasileiras com regime hidrológico natural, avaliando a
    transferibilidade dos parâmetros físicos via *encoder* neural de
    atributos fisiográficos.
    """
)

tab_motivacao, tab_etapas, tab_cronograma = st.tabs([
    "Motivação e desenho experimental",
    "Etapas",
    "Cronograma",
])

# ----------------------------------------------------------- Motivação / PUB
with tab_motivacao:
    st.subheader("Da bacia única à regionalização multi-bacia")
    st.markdown(
        r"""
        Os resultados da Fase 1 fornecem evidências favoráveis à
        viabilidade da predição em bacias não monitoradas
        (*Prediction in Ungauged Basins* — PUB). Três elementos
        fundamentam a extensão multi-bacia:

        1. O modelo `LSTM_TTD_Base_Fixed` alcançou NSE = 0,82 em
           simulação contínua com parâmetros físicos padrão da
           literatura ($\lambda = 0{,}2$, $t_{c\_scale} = 1{,}0$,
           $\sigma = 3{,}0$ h), indicando que valores razoáveis
           derivados de atributos da bacia produzem desempenho
           robusto sem calibração local.
        2. Os parâmetros convergidos pelos modelos ajustáveis
           ($t_{c\_scale} \approx 1{,}2$–1,3; $\sigma \approx 4$–5 h;
           $\lambda \approx 0{,}06$–0,17) mantêm interpretação
           física compatível com a literatura brasileira e com a
           geomorfologia da bacia — condição necessária para que
           sejam regionalizáveis.
        3. O componente de maior impacto nos resultados é o TTD
           (ganho de até 35% em relação ao modelo de referência),
           cujo parâmetro principal $T_c$ pode ser calculado
           diretamente do DEM para qualquer bacia do território
           nacional.

        #### Hipótese central da Fase 2

        Os parâmetros físicos aprendíveis podem ser preditos por um
        *encoder* neural (MLP) a partir de atributos da bacia,
        permitindo generalização espacial para bacias não
        monitoradas:
        """
    )

    st.latex(r"[\,t_{c\_scale},\;\sigma,\;\lambda\,] = \mathrm{MLP}(\mathbf{atributos}_{\text{bacia}})")

    st.markdown(
        """
        #### Atributos fisiográficos candidatos

        A metodologia adota atributos disponíveis para qualquer
        bacia brasileira, dispensando estação fluviométrica local:

        - Área de drenagem e densidade de drenagem;
        - Elevação média e declividade média (ANADEM);
        - Curve Number médio (produto BHAE_CN-2022 da ANA);
        - Frações de uso do solo (MapBiomas Coleção 8.0);
        - Precipitação média anual (MERGE/CPTEC);
        - Índice de aridez.

        #### Desenho experimental

        A validação adota o protocolo *leave-one-basin-out* como
        teste principal, complementado por *k-fold* espacial. Em
        ambos os casos, o modelo é avaliado em bacias excluídas do
        treinamento, simulando a aplicação em bacias não
        monitoradas. A comparação com modelos de referência externos
        inclui o EA-LSTM (Kratzert et al., 2019) e o $\\delta$HBV
        (Feng et al., 2024), contextualizando o desempenho no estado
        da arte internacional.
        """
    )

# --------------------------------------------------------------------- Etapas
with tab_etapas:
    st.subheader("Etapas metodológicas da Fase 2")
    st.markdown(
        """
        A Fase 2 é organizada em quatro etapas sequenciais, com
        dependência entre fases: a seleção de bacias condiciona a
        preparação dos dados, que por sua vez condiciona o
        treinamento e a validação espacial.
        """
    )

    st.markdown(
        """
        #### Etapa 1 — Seleção de bacias

        Identificação de aproximadamente 100 bacias brasileiras com
        séries horárias de vazão disponíveis via ANA/HidroWeb,
        cobertura MERGE para precipitação e regime hidrológico
        predominantemente natural (ausência de reservatórios de
        grande porte a montante). O conjunto deve contemplar
        distribuição geográfica representativa dos principais biomas
        e regimes climáticos do território nacional. Uma
        infraestrutura automatizada de aquisição via HidroWebService
        já foi desenvolvida e encontra-se em uso.
        """
    )

    st.markdown(
        """
        #### Etapa 2 — Preparação de dados multi-bacia

        Consolidação de banco de dados regional com precipitação
        MERGE horária, vazão telemétrica ANA e atributos
        fisiográficos por bacia (CN do BHAE, uso do solo do
        MapBiomas, topografia do ANADEM), seguindo a estrutura do
        CAMELS-BR para compatibilidade com estudos internacionais.
        O pré-processamento inclui discretização em ottobacias,
        cálculo de $T_c$ distribuído por bacia e controle de
        qualidade das séries telemétricas.
        """
    )

    st.markdown(
        """
        #### Etapa 3 — Treinamento multi-bacia com *encoder*

        Treinamento conjunto da arquitetura TTD-SCS-LSTM acoplada a
        uma MLP (*encoder*) que mapeia atributos fisiográficos aos
        parâmetros físicos aprendíveis. O treinamento é
        *end-to-end* e diferenciável, com otimização simultânea dos
        pesos da LSTM, da MLP e dos parâmetros residuais. A
        configuração preserva o princípio da Fase 1: previsão
        baseada exclusivamente em precipitação e atributos, sem
        vazão observada como entrada.
        """
    )

    st.markdown(
        """
        #### Etapa 4 — Validação espacial e análise por bioma

        Avaliação do desempenho via *leave-one-basin-out* e *k-fold*
        espacial, com reporte de NSE, KGE, PBIAS e intervalos de
        confiança por múltiplas sementes. A análise final inclui
        estratificação por bioma/região climática, identificação
        dos atributos mais informativos para generalização e
        comparação sistemática com EA-LSTM e $\\delta$HBV nas mesmas
        bacias.
        """
    )

# ---------------------------------------------------------------- Cronograma
with tab_cronograma:
    st.subheader("Cronograma — da qualificação à defesa")
    st.markdown(
        """
        A conclusão do doutorado está prevista para aproximadamente
        22 meses após a qualificação (abril/2026). A sequência
        articula o fechamento da Fase 1 (Artigo 1), a execução
        integral da Fase 2 (Artigo 2), a redação da tese e a defesa.
        """
    )

    tasks = [
        ("Finalização da quali e seleção de bacias",
         "2026-05-01", "2026-05-31", "Preparação"),
        ("Preparação de dados multi-bacia (MERGE + ANA + CAMELS-BR)",
         "2026-06-01", "2026-08-31", "Preparação"),
        ("Submissão Artigo 1 (metodologia Fase 1)",
         "2026-06-01", "2026-06-30", "Publicações"),
        ("Treinamento multi-bacia e leave-one-basin-out",
         "2026-09-01", "2026-12-31", "Fase 2"),
        ("Análise de resultados e redação do Artigo 2",
         "2027-01-01", "2027-03-31", "Fase 2"),
        ("Submissão Artigo 2 (regionalização)",
         "2027-04-01", "2027-04-30", "Publicações"),
        ("Redação e revisões da tese",
         "2027-07-01", "2027-12-31", "Tese"),
        ("Defesa da tese",
         "2028-03-01", "2028-03-31", "Tese"),
    ]

    df_tasks = pd.DataFrame(tasks, columns=["Atividade", "Início", "Fim", "Categoria"])
    df_tasks["Início"] = pd.to_datetime(df_tasks["Início"])
    df_tasks["Fim"] = pd.to_datetime(df_tasks["Fim"])

    color_map = {
        "Preparação": "#f59e0b",
        "Fase 2": "#9333ea",
        "Publicações": "#2563eb",
        "Tese": "#dc2626",
    }

    fig = px.timeline(
        df_tasks,
        x_start="Início",
        x_end="Fim",
        y="Atividade",
        color="Categoria",
        color_discrete_map=color_map,
        hover_data={"Início": "|%b/%Y", "Fim": "|%b/%Y", "Categoria": True},
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(
        title="Janela temporal",
        tickformat="%b/%Y",
        dtick="M3",
        gridcolor="#f1f5f9",
    )
    fig.update_layout(
        template="plotly_white",
        height=440,
        margin=dict(l=20, r=20, t=10, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title=None,
        ),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "**Leitura do cronograma.** A cor das barras identifica a natureza "
        "da atividade: preparação de dados (âmbar), execução da Fase 2 "
        "(roxo), submissões de artigos (azul) e redação/defesa da tese "
        "(vermelho). A Fase 2 ocupa o intervalo de setembro/2026 a "
        "março/2027, seguida pela submissão do Artigo 2 em abril/2027. "
        "A defesa está programada para o primeiro semestre de 2028."
    )

    st.divider()

    st.markdown(
        """
        #### Artigos planejados

        | Artigo | Título | Periódico-alvo | Submissão |
        |---|---|---|---|
        | 1 | *A Differentiable Hydrological Framework Integrating Distributed Unit Hydrograph, SCS-CN, and LSTM for Hourly Streamflow Prediction* | *Journal of Hydrology* ou *Water Resources Research* | Junho/2026 |
        | 2 | *Prediction in Ungauged Basins via Regionalized Hydrological Parameters: A Multi-Catchment Differentiable Framework for Brazil* | *Journal of Hydrology* ou *HESS* | Abril/2027 |
        """
    )
