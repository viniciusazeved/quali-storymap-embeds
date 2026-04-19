"""
Widget: revisao bibliografica sintetica — Capitulo 2 da qualificacao.

Replicado a partir do app principal (pagina "6. Revisao bibliografica"),
com tom academico ajustado para consumo no StoryMap. Seis eixos em tabs:
modelos conceituais, LSTM em hidrologia, arquiteturas temporais,
modelos diferenciaveis, SCS-CN em hibridos, sintese/posicionamento.

URL: /revisao_bibliografica
"""
from __future__ import annotations

import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Revisão bibliográfica",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

st.markdown(
    """
    Síntese do Capítulo 2 da qualificação, organizada em seis eixos:
    modelos conceituais, redes LSTM em hidrologia, arquiteturas temporais
    alternativas, modelos hidrológicos diferenciáveis, SCS-CN em modelos
    híbridos, e posicionamento do trabalho proposto em relação ao estado
    da arte.
    """
)

tab_conc, tab_lstm, tab_arch, tab_diff, tab_hib, tab_sint = st.tabs([
    "Modelos conceituais",
    "LSTM em hidrologia",
    "Arquiteturas temporais",
    "Modelos diferenciáveis",
    "SCS-CN em híbridos",
    "Síntese e posicionamento",
])

# --------------------------------------------------------------- Conceituais
with tab_conc:
    st.subheader("Modelos conceituais — referência operacional")
    st.markdown(
        """
        Modelos conceituais representam a bacia como sistema de reservatórios
        interconectados, em que cada reservatório simula um processo
        hidrológico específico (interceptação, armazenamento no solo,
        escoamento superficial e subterrâneo).

        **Referências consolidadas na literatura internacional:**

        | Modelo | Origem | Características |
        |---|---|---|
        | GR4J | França (Perrin et al., 2003) | Quatro parâmetros; parcimônia e desempenho em diversos climas |
        | HBV | Suécia (Lindström et al., 1997) | Representação explícita de neve; uso extensivo em regiões temperadas |
        | Sacramento (SAC-SMA) | EUA (Burnash, 1973) | Dezesseis parâmetros; operacional no National Weather Service |
        | MGB | Brasil (Collischonn et al., 2007) | Semi-distribuído com conceito VIC; adaptado a bacias brasileiras |

        **Limitações reportadas na literatura:**

        1. *Equifinalidade* (Beven, 2001): distintas combinações de parâmetros
           produzem ajustes estatisticamente equivalentes aos dados observados.
        2. Dependência de vazão observada para calibração, limitando a
           aplicação em bacias não monitoradas.
        3. Estrutura rígida com reservatórios e equações pré-definidas.
        4. Predominância de formulação concentrada (*lumped*), que não
           preserva a heterogeneidade espacial da bacia.

        O método SCS-CN (Soil Conservation Service, 1972) permanece como
        uma das ferramentas mais utilizadas para separação de escoamento
        sem exigir calibração com dados de vazão. O valor clássico
        λ = 0,2 foi questionado por Woodward et al. (2003) e por
        ValleJunior et al. (2019), que reportaram mediana de λ = 0,045
        em bacias tropicais brasileiras — motivação para tratá-lo como
        parâmetro aprendível neste trabalho.
        """
    )

# --------------------------------------------------------------------- LSTM
with tab_lstm:
    st.subheader("Redes LSTM em hidrologia")
    st.markdown(
        """
        As redes *Long Short-Term Memory* (Hochreiter & Schmidhuber, 1997)
        constituem uma classe de redes neurais recorrentes que processam
        séries temporais com mecanismo explícito de portões (entrada,
        esquecimento, saída), permitindo preservar informação relevante
        ao longo de horizontes temporais extensos. Sua aplicação em
        modelagem chuva-vazão consolidou-se a partir de 2018.
        """
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            **Contribuições seminais:**

            - Kratzert et al. (2018): LSTMs treinadas em múltiplas bacias
              (CAMELS-US) superam modelos conceituais como o SAC-SMA,
              com NSE mediano de 0,63 em validação temporal.
            - Kratzert et al. (2019): EA-LSTM (*Entity-Aware LSTM*)
              incorpora atributos estáticos da bacia; NSE mediano
              0,76–0,88 em regionalização CAMELS-US.
            - Frame et al. (2022): LSTMs multi-bacia reproduzem eventos
              extremos com acurácia superior à de modelos conceituais
              calibrados individualmente.
            - Lees et al. (2022): estados internos da LSTM apresentam
              correspondência com variáveis hidrológicas (umidade do
              solo, neve) mesmo sem supervisão explícita.
            """
        )

    with col_b:
        st.markdown(
            """
            **Limitações reportadas:**

            - Baixa interpretabilidade dos estados internos em relação
              a variáveis observáveis (Reichstein et al., 2019).
            - Degradação potencial em condições fora da distribuição
              de treinamento (Frame et al., 2022).
            - Ausência de garantias de conservação de massa (Hoedt et
              al., 2021).
            - Opacidade limita adoção em decisões operacionais
              críticas (Nearing et al., 2021).

            Nearing et al. (2021) propõem realocar o papel do hidrólogo
            para o fornecimento de *conhecimento prévio estrutural* que
            guie o aprendizado — ponto de partida dos modelos
            hidrológicos diferenciáveis.
            """
        )

    st.markdown(
        """
        Kratzert et al. (2024) consolidam a recomendação de que LSTMs não
        devem ser treinadas em bacia única: o treinamento multi-bacia
        força a rede a aprender padrões gerais em detrimento de
        características locais — premissa que orienta a Fase 2 desta
        pesquisa, em regionalização para ~100 bacias brasileiras.
        """
    )

# ------------------------------------------------------------- Arquiteturas
with tab_arch:
    st.subheader("Arquiteturas temporais — LSTM, Transformer, TCN, Mamba")
    st.markdown(
        """
        Três famílias de arquiteturas dominam a literatura recente. A
        convergência de desempenho observada em *benchmarks* (Liu et al.,
        2025) reforça que a escolha do *encoder* temporal é menos crítica
        que a qualidade da representação física e dos dados de entrada.

        | Característica | LSTM | Transformer | TCN | Mamba |
        |---|---|---|---|---|
        | Processamento | Sequencial | Paralelo | Paralelo | Paralelo |
        | Complexidade | $O(n)$ | $O(n^2)$ | $O(n \\log n)$ | $O(n)$ |
        | Memória | Célula explícita | Atenção | Campo receptivo fixo | Estado estruturado |
        | Volume de dados | Moderado | Grande | Moderado | Moderado |
        | Integração com física | Estabelecida | Emergente | Limitada | Exploratória |
        | Aplicações em hidrologia | Extensivas | Crescentes | Limitadas | Incipientes |

        **Justificativa da escolha da LSTM neste trabalho:**

        1. Histórico extenso de validação em hidrologia desde Kratzert
           (2018, 2019), consolidando-se como referência de comparação.
        2. Compatibilidade estabelecida com camadas físicas diferenciáveis
           (δHBV, DeepGR4J).
        3. Estados internos com correspondência empírica a variáveis
           hidrológicas (Lees et al., 2022), o que facilita análise
           post-hoc do comportamento do modelo.

        Experimentos exploratórios conduzidos na fase anterior da pesquisa
        (Tensor Hydro, 2025) avaliaram o uso de Mamba como *encoder*
        temporal. Os resultados não apresentaram ganho significativo,
        consistente com a convergência entre arquiteturas reportada em
        Liu et al. (2025).
        """
    )

# ----------------------------------------------------------- Diferenciáveis
with tab_diff:
    st.subheader("Modelos hidrológicos diferenciáveis")
    st.markdown(
        """
        A classificação proposta por Shen et al. (2023) e Xu et al. (2025)
        identifica quatro estratégias para integrar física e aprendizado
        de máquina em hidrologia:

        **1. Predição neural de parâmetros físicos.** Uma rede neural estima
        os parâmetros de um modelo conceitual a partir de atributos da
        bacia. Tsai et al. (2021) demonstraram a viabilidade para o
        SAC-SMA, prescindindo da calibração individual por bacia. Esta
        é a linha de trabalho adotada na Fase 2 desta pesquisa.

        **2. Substituição de submódulos conceituais por redes neurais.**
        Componentes específicos do modelo (ex.: propagação de escoamento)
        são substituídos por redes. He et al. (2025) avaliaram essa
        estratégia no HBV, indicando maior contribuição da substituição
        do módulo de propagação em relação ao módulo de geração.

        **3. Correção neural de erros residuais.** A rede neural atua como
        pós-processador do modelo físico, corrigindo erros sistemáticos
        (Frame et al., 2022). Aproxima-se do papel atribuído à LSTM no
        modelo proposto.

        **4. Otimização *end-to-end* com gradientes.** O modelo integral
        — componentes físicos e neurais — é expresso como grafo
        diferenciável. Os parâmetros físicos são ajustados por
        retropropagação conjuntamente com os pesos da rede (Feng et
        al., 2022).

        **Implementações representativas:**

        - δHBV (Feng et al., 2022, 2024; Song et al., 2025): HBV
          totalmente diferenciável; Ji et al. (2025, *Nature
          Communications*) reportam padrões hidrológicos globais com
          parâmetros interpretáveis em escala continental.
        - DeepGR4J (Kapoor et al., 2023): GR4J integrado a
          CNN/LSTM, avaliado em 223 bacias australianas.
        - DRRAiNN (Scholz et al., 2025): modelo distribuído
          totalmente diferenciável com rastreamento de fontes.
        - TS-DUH (Hu et al., 2024): campo de velocidades de Maidment
          com SCS-CN, com parâmetros fixos e sem componente neural
          — não diferenciável.

        A revisão não identificou formulação diferenciável do campo de
        velocidades de Maidment (1996) como camada neural com parâmetros
        aprendíveis. Os trabalhos mais próximos empregam distribuições
        gamma paramétricas sem vinculação direta à geomorfologia da
        bacia. Esta constitui a primeira lacuna preenchida pelo presente
        trabalho.
        """
    )

# --------------------------------------------------------- SCS-CN híbridos
with tab_hib:
    st.subheader("SCS-CN em modelos híbridos")
    st.markdown(
        """
        Isik et al. (2013) desenvolveram um dos primeiros modelos
        combinando SCS-CN com rede neural artificial. Merizalde et al.
        (2023) empregaram SCS-CN como pré-processador de LSTMs e Meer et
        al. (2025) revisam abordagens análogas.

        A revisão conduzida não identificou implementação do SCS-CN como
        componente totalmente diferenciável com o coeficiente de abstração
        inicial $\\lambda$ aprendível via retropropagação e CN mantido
        fixo como conhecimento prévio. Os trabalhos existentes adotam
        uma abordagem modular: as saídas do SCS-CN alimentam a rede
        neural como variáveis de entrada, mas os gradientes não fluem
        através das equações do método. Esta constitui a segunda lacuna
        preenchida pelo presente trabalho.

        #### Variabilidade empírica de $\\lambda$

        - Banasik et al. (2014): $\\lambda$ predominantemente inferior a
          0,05 em bacia urbanizada polonesa.
        - Afrasiabikia et al. (2025): $\\lambda$ variando entre 0,01 e
          0,30; calibração conjunta com CN melhora NSE (0,78) em relação
          ao valor fixo.
        - ValleJunior et al. (2019) e Oliveira et al. (2016): em bacias
          tropicais brasileiras, $\\lambda$ mediano = 0,045, com 96,7%
          dos valores inferiores a 0,2.

        A fixação de $\\lambda = 0{,}2$ introduz erro sistemático em
        aplicações em bacias tropicais brasileiras. O tratamento de
        $\\lambda$ como parâmetro aprendível, com CN fixo proveniente do
        produto oficial BHAE_CN-2022 da ANA, preserva consistência com
        o dado nacional e viabiliza a transferência para bacias não
        monitoradas na Fase 2.
        """
    )

# --------------------------------------------------------------- Síntese
with tab_sint:
    st.subheader("Síntese — lacunas identificadas")
    st.markdown(
        """
        A revisão identifica três lacunas específicas que orientam a
        formulação do modelo proposto:

        #### Lacuna 1 — TTD diferenciável baseado em Maidment

        Trabalhos recentes em modelagem diferenciável ($\\delta$HBV,
        DeepGR4J) empregam distribuições gamma paramétricas como
        distribuição de tempos de viagem. A formulação baseada no campo
        de velocidades de Maidment (1996), cujos parâmetros mantêm
        correspondência direta com grandezas geomorfológicas mensuráveis
        ($t_{c\\_scale}$, $\\sigma$), não foi identificada como camada
        diferenciável na literatura revisada.

        #### Lacuna 2 — SCS-CN com $\\lambda$ aprendível

        Embora $\\lambda$ seja reconhecidamente dependente de condições
        locais (Brandão et al., 2025), os estudos revisados não reportam
        sua implementação como variável aprendível em camada neural
        diferenciável.

        #### Lacuna 3 — Integração tripla TTD + SCS-CN + LSTM

        Não foi identificada arquitetura diferenciável *end-to-end* que
        integre simultaneamente (i) hidrograma unitário distribuído
        baseado em tempo de viagem, (ii) separação de escoamento via
        SCS-CN e (iii) rede LSTM para refinamento residual.

        ---

        #### Posicionamento em relação ao estado da arte

        | Aspecto | Abordagens existentes | Modelo proposto |
        |---|---|---|
        | Hidrograma unitário | Modelos concentrados (Nash, Clark) ou Maidment com parâmetros fixos | IUH gaussiano com $t_{c\\_scale}$, $\\sigma$ aprendíveis |
        | Separação de escoamento | SCS-CN tabelado ou calibrado por otimização global | CN fixo (BHAE-ANA) e $\\lambda$ aprendível |
        | Papel da rede neural | Modelar hidrograma completo ou substituir módulo conceitual | Refinamento residual da estimativa física |
        | Formulação de saída | $Q = \\mathrm{NN}(\\mathrm{inputs})$ ou modelo conceitual puro | $Q = \\mathrm{LSTM}(Q_{\\text{physics}}, P, t)$ |
        | Diferenciabilidade | Total em recentes ($\\delta$HBV, DeepGR4J) com IUH gamma | Total com IUH gaussiano vinculado a Maidment |
        | Aplicabilidade em PUB | Requer dados locais ou degrada em extrapolação | Arquitetura formulada para PUB (não autorregressiva, parâmetros regionalizáveis) |

        A combinação TTD de Maidment + SCS-CN + LSTM em arquitetura
        *end-to-end*, com o papel da LSTM explicitamente definido como
        correção residual, não foi identificada na literatura revisada
        (2018–2026).
        """
    )

st.divider()

with st.expander("Referências principais por tema"):
    st.markdown(
        """
        | Tema | Referências principais |
        |---|---|
        | Hidrologia clássica | Tucci (2005); Beven (2012); Beven (2001) |
        | Hidrograma unitário | Sherman (1932); Clark (1945); Nash (1957); Maidment et al. (1996) |
        | SCS-CN | SCS (1972); Woodward et al. (2003); ValleJunior et al. (2019); Afrasiabikia et al. (2025) |
        | Arquiteturas temporais | Hochreiter & Schmidhuber (1997); Vaswani et al. (2017); Gu & Dao (2024) |
        | LSTM em hidrologia | Kratzert et al. (2018, 2019, 2024); Frame et al. (2022); Lees et al. (2022); Nearing et al. (2021) |
        | Modelos diferenciáveis | Feng et al. (2022, 2024); Shen et al. (2023); Kapoor et al. (2023); Xu et al. (2025) |
        | Restrições físicas | Karniadakis et al. (2021); Frame et al. (2023); Pokharel et al. (2023) |
        | Explicabilidade | Lundberg & Lee (2017); Lees et al. (2022); Hu et al. (2025) |
        | PUB e regionalização | Sivapalan (2003); Blöschl et al. (2013); Hrachowitz et al. (2013) |
        """
    )
