"""
Widget: metodologia_short — versao enxuta para apresentacao ao vivo.

Apenas os tres modulos (SCS-CN, TTD, LSTM) com simuladores interativos
e textos curtos. Fonte ampliada para legibilidade em projecao.

URL: /metodologia_short
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _embed_css import hide_streamlit_chrome

st.set_page_config(
    page_title="Metodologia · Módulos",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_streamlit_chrome()

# --------------------------------------------------------------------------
# CSS — fonte maior, espaçamento generoso, pensado para apresentação
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      html, body, [class*="css"], .main * {
        font-size: 18px !important;
      }
      .main h1 { font-size: 36px !important; font-weight: 700 !important; }
      .main h2 { font-size: 28px !important; font-weight: 600 !important; }
      .main h3 { font-size: 22px !important; font-weight: 600 !important; }
      .stTabs [data-baseweb="tab"] {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 12px 20px !important;
      }
      .stMarkdown p { line-height: 1.6 !important; }
      .stSlider label, .stRadio label, .stSelectbox label {
        font-size: 18px !important;
        font-weight: 500 !important;
      }
      .stMetric label { font-size: 16px !important; }
      .stMetric [data-testid="stMetricValue"] { font-size: 28px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

st.markdown(
    """
    ## Metodologia · módulos, avaliação e extensão multi-bacia

    Síntese das seções 3.4 a 3.10 da qualificação: os três módulos da
    arquitetura, o estudo comparativo, o protocolo de treinamento e
    avaliação, os dois produtos operacionais de previsão e a estratégia
    de regionalização para bacias não monitoradas (Fase 2).
    """
)

tab_scs, tab_ttd, tab_lstm, tab_abl, tab_eval, tab_reg = st.tabs([
    "SCS-CN",
    "TTD",
    "LSTM",
    "Estudo comparativo",
    "Treinamento e avaliação",
    "Regionalização (Fase 2)",
])


# ==========================================================================
#  SCS-CN
# ==========================================================================
with tab_scs:
    st.markdown(
        r"""
        ### Módulo SCS-CN — separação de escoamento

        Converte a precipitação total $P$ em precipitação efetiva $P_e$
        (parcela que escoa), em função do *Curve Number* ($CN$) e do
        coeficiente de abstração inicial $\lambda$. Na nossa arquitetura,
        $\lambda$ é **aprendível** por retropropagação; o $CN$ é fixo
        (produto BHAE_CN-2022 da ANA).
        """
    )

    st.latex(r"P_e = \frac{\mathrm{ReLU}(P - \lambda S)^2}{\mathrm{ReLU}(P - \lambda S) + S + \epsilon}, \quad S = 25{,}4\left(\frac{1000}{CN} - 10\right)")

    c1, c2, c3 = st.columns(3)
    with c1:
        cn = st.slider("CN", 30, 100, 63, 1,
                       help="Bacia do rio Preto: CN médio = 63,0")
    with c2:
        lam = st.slider("λ", 0.01, 0.40, 0.20, 0.01,
                        help="Clássico: 0,20 · bacias tropicais brasileiras ~0,045")
    with c3:
        p_max = st.slider("P máx (mm)", 10, 200, 80, 10)

    P = np.linspace(0, p_max, 200)
    S = 25.4 * (1000 / cn - 10)
    relu_arg = np.maximum(P - lam * S, 0)
    Pe = relu_arg ** 2 / (relu_arg + S + 1e-6)

    fig_scs = go.Figure()
    fig_scs.add_trace(go.Scatter(
        x=P, y=Pe, mode="lines",
        line=dict(color=PALETTE[0], width=4),
        name="Precipitação efetiva (P_e)",
    ))
    fig_scs.add_trace(go.Scatter(
        x=P, y=P, mode="lines",
        line=dict(color="#94a3b8", dash="dash", width=2),
        name="P_e = P (escoamento total)",
    ))
    fig_scs.add_vline(
        x=lam * S, line_dash="dot", line_color=PALETTE[1],
        annotation_text=f"λS = {lam * S:.1f} mm",
        annotation_font=dict(size=14),
    )
    fig_scs.update_layout(
        title=dict(
            text=f"CN = {cn} · λ = {lam:.2f} · S = {S:.1f} mm",
            font=dict(size=20),
        ),
        template="plotly_white",
        xaxis=dict(title="Precipitação P (mm)", title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(title="Precipitação efetiva P_e (mm)", title_font=dict(size=16), tickfont=dict(size=14)),
        height=440,
        legend=dict(font=dict(size=14), yanchor="top", y=0.98, xanchor="left", x=0.02),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    st.plotly_chart(fig_scs, use_container_width=True)


# ==========================================================================
#  TTD — hidrograma unitário distribuído
# ==========================================================================
with tab_ttd:
    st.markdown(
        r"""
        ### Módulo TTD — hidrograma unitário distribuído

        Propaga a precipitação efetiva até o exutório por convolução com
        um hidrograma unitário gaussiano, um para cada ottobacia. Os
        parâmetros são o tempo de concentração $T_c$ (calculado do DEM,
        com fator de escala aprendível) e a dispersão $\sigma$
        (aprendível). O conjunto das 245 respostas compõe o hidrograma
        unitário distribuído.
        """
    )

    st.latex(r"h(t) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{(t - T_c)^2}{2\sigma^2}\right]")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        tc_values = st.multiselect(
            "Tempos de concentração T_c (h)",
            options=[0.5, 1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 76],
            default=[3, 12, 24, 48],
            help="Bacia do rio Preto: Tc médio (Manning) = 23,4 h · máx = 76,6 h",
        )
    with col_b:
        sigma = st.slider("σ (h)", 0.5, 12.0, 3.0, 0.5,
                          help="Valor inicial no treinamento: 3,0 h")

    if tc_values:
        t = np.linspace(0, 96, 400)
        fig_ttd = go.Figure()
        for i, tc in enumerate(tc_values):
            h = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((t - tc) ** 2) / (2 * sigma ** 2))
            fig_ttd.add_trace(go.Scatter(
                x=t, y=h,
                mode="lines",
                name=f"Tc = {tc:.1f} h",
                line=dict(color=PALETTE[i % len(PALETTE)], width=3),
            ))
        fig_ttd.update_layout(
            title=dict(text=f"IUH gaussiano · σ = {sigma:.1f} h", font=dict(size=20)),
            template="plotly_white",
            xaxis=dict(title="Tempo (h)", title_font=dict(size=16), tickfont=dict(size=14)),
            yaxis=dict(title="h(t)", title_font=dict(size=16), tickfont=dict(size=14)),
            height=440,
            legend=dict(font=dict(size=14)),
            margin=dict(l=60, r=30, t=50, b=50),
        )
        st.plotly_chart(fig_ttd, use_container_width=True)
    else:
        st.info("Selecione ao menos um valor de Tc.")


# ==========================================================================
#  LSTM
# ==========================================================================
with tab_lstm:
    st.markdown(
        r"""
        ### Módulo LSTM — refinamento neural

        A LSTM recebe como entrada a vazão física estimada pelo TTD
        ($Q_\text{physics}$), a precipitação ($P$) e variáveis temporais
        ($t$). A rede **não gera a vazão do zero**: ela corrige a
        estimativa física, capturando padrões não representados pelos
        módulos simplificados, como umidade antecedente e fluxo de base.
        O simulador abaixo ilustra esse papel.
        """
    )

    cl1, cl2 = st.columns(2)
    with cl1:
        peso = st.slider(
            "Intensidade da correção da LSTM",
            0.0, 1.0, 0.6, 0.05,
            help="0 = modelo apenas físico (Q = Q_physics) · 1 = correção plena",
        )
    with cl2:
        atraso = st.slider(
            "Defasagem do Q_physics (h)",
            0, 12, 4, 1,
            help="Ilustra a subestimação temporal que a LSTM precisa corrigir",
        )

    # Hidrograma sintético: evento de cheia
    t = np.linspace(0, 240, 480)
    tc_evento = 100.0
    sig_evento = 14.0
    Q_obs = 280.0 * np.exp(-((t - tc_evento) ** 2) / (2 * sig_evento ** 2)) + 30.0
    Q_physics = 0.80 * 280.0 * np.exp(-((t - tc_evento - atraso) ** 2) / (2 * (sig_evento * 1.1) ** 2)) + 30.0
    Q_pred = Q_physics + peso * (Q_obs - Q_physics)

    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(
        x=t, y=Q_obs, mode="lines",
        line=dict(color="#0f172a", width=3),
        name="Q observado (referência)",
    ))
    fig_lstm.add_trace(go.Scatter(
        x=t, y=Q_physics, mode="lines",
        line=dict(color="#94a3b8", dash="dash", width=2.5),
        name="Q_physics (saída do TTD)",
    ))
    fig_lstm.add_trace(go.Scatter(
        x=t, y=Q_pred, mode="lines",
        line=dict(color=PALETTE[0], width=3.5),
        name="Q previsto (após LSTM)",
    ))
    fig_lstm.update_layout(
        title=dict(
            text=f"Refinamento neural · intensidade = {peso:.2f}",
            font=dict(size=20),
        ),
        template="plotly_white",
        xaxis=dict(title="Tempo (h)", title_font=dict(size=16), tickfont=dict(size=14)),
        yaxis=dict(title="Vazão (m³/s)", title_font=dict(size=16), tickfont=dict(size=14)),
        height=440,
        legend=dict(font=dict(size=14), yanchor="top", y=0.98, xanchor="left", x=0.02),
        margin=dict(l=60, r=30, t=50, b=50),
    )
    st.plotly_chart(fig_lstm, use_container_width=True)

    st.caption(
        "Exemplo ilustrativo com dados sintéticos. O simulador ajusta um "
        "fator único de correção para fins didáticos; no modelo real, a "
        "correção é aprendida ponto a ponto pela LSTM com lookback de 240 h "
        "(2 camadas, hidden = 64, dropout = 0,1)."
    )


# ==========================================================================
#  Estudo comparativo (ablação) — 10 configurações
# ==========================================================================
with tab_abl:
    st.markdown(
        """
        ### Estudo comparativo — 10 configurações

        Para isolar a contribuição de cada módulo, o estudo compara
        sistematicamente dez configurações do modelo, variando a
        discretização espacial (concentrada vs. distribuída), a
        formulação do tempo de concentração (Base vs. Manning), a
        presença do SCS-CN e o regime dos parâmetros físicos (fixos
        vs. aprendíveis).
        """
    )

    st.markdown(
        """
        | # | Modelo | Espacial | Tc | SCS-CN | Parâmetros |
        |:-:|:--|:--|:--|:-:|:--|
        | 1 | LSTM_Lumped | Concentrado | — | — | — |
        | 2 | LSTM | Distribuído | — | — | — |
        | 3 | LSTM_TTD_Base_Fixed | Distribuído | Base (Maidment) | — | Fixos |
        | 4 | LSTM_TTD_Base | Distribuído | Base (Maidment) | — | Aprendíveis |
        | 5 | LSTM_TTD_Manning_Fixed | Distribuído | Manning | — | Fixos |
        | 6 | LSTM_TTD_Manning | Distribuído | Manning | — | Aprendíveis |
        | 7 | LSTM_TTD_Base_SCS_Fixed | Distribuído | Base (Maidment) | Sim | Fixos |
        | 8 | LSTM_TTD_Base_SCS | Distribuído | Base (Maidment) | Sim | Aprendíveis |
        | 9 | LSTM_TTD_Manning_SCS_Fixed | Distribuído | Manning | Sim | Fixos |
        | 10 | LSTM_TTD_Manning_SCS | Distribuído | Manning | Sim | Aprendíveis |
        """
    )

    st.markdown(
        """
        #### Cinco hipóteses operacionais

        - **H1** — Distribuído ($>$) concentrado.
        - **H2** — Tc Manning ($>$) Tc Base.
        - **H3** — Parâmetros aprendíveis ($>$) fixos.
        - **H4** — SCS-CN agrega valor quando combinado com o TTD.
        - **H5** — Modelo híbrido ($>$) LSTM puro e configuração concentrada.

        Os dois modos de avaliação, previsão multi-horizonte e simulação
        contínua, são aplicados às dez configurações, formando a base do
        capítulo de resultados.
        """
    )


# ==========================================================================
#  Treinamento e Avaliação (§3.8 + §3.9)
# ==========================================================================
with tab_eval:
    st.markdown(
        """
        ### Configuração experimental

        A divisão temporal é cronológica, sem embaralhamento, preservando
        a estrutura temporal dos dados e simulando a aplicação operacional
        do modelo.

        | Conjunto | Período | Proporção |
        |:--|:--|:-:|
        | Treino | 2021-01-01 a 2024-06-29 | ≈ 70 % |
        | Validação (*early stopping*) | 2024-06-29 a 2025-03-30 | ≈ 15 % |
        | Teste | 2025-03-30 a 2025-12-30 | ≈ 15 % |

        Todos os dez modelos do estudo comparativo foram treinados com
        configuração idêntica, de modo que as diferenças de desempenho
        sejam atribuíveis exclusivamente à arquitetura.
        """
    )

    col_hp, col_mt = st.columns(2)
    with col_hp:
        st.markdown(
            """
            #### Hiperparâmetros de treinamento

            | Parâmetro | Valor |
            |:--|:-:|
            | Otimizador | Adam |
            | *Learning rate* | 10⁻³ |
            | Épocas máximas | 300 |
            | *Early stopping* | 30 épocas |
            | *Batch size* | 1.024 |
            | GPU | NVIDIA RTX 3000 Ada |
            """
        )

    with col_mt:
        st.markdown(
            """
            #### Métricas e classificação

            Métrica principal: NSE (Nash–Sutcliffe).
            Complementares: KGE, PBIAS, RMSE.

            | Classificação | NSE | PBIAS (%) |
            |:--|:-:|:-:|
            | Muito Bom | > 0,75 | < ±10 |
            | Bom | 0,65–0,75 | ±10 a ±15 |
            | Satisfatório | 0,50–0,65 | ±15 a ±25 |
            | Insatisfatório | < 0,50 | > ±25 |

            *Fonte: Moriasi et al. (2007).*
            """
        )

    st.divider()

    st.markdown(
        """
        ### Dois produtos de previsão hidrológica

        A mesma arquitetura gera dois produtos operacionais distintos,
        avaliados separadamente.
        """
    )

    st.markdown(
        """
        | Característica | Previsão multi-horizonte | Simulação contínua |
        |:--|:--|:--|
        | Horizonte temporal | 1, 3, 6, 12, 24 h | Meses a anos |
        | Tipo de saída | Vetor multi-horizonte | Série temporal completa |
        | Frequência de atualização | A cada hora | Única (período completo) |
        | Dependência de dados recentes | Alta | Baixa |
        | Aplicação principal | Alerta hidrológico, defesa civil, operação em tempo real | Disponibilidade hídrica, outorga, regionalização |
        | Métrica principal | NSE por horizonte | NSE da série completa |
        """
    )

    st.caption(
        "A avaliação nos dois contextos é essencial para garantir robustez: "
        "bom desempenho apenas em previsão pode indicar captura de padrões "
        "de persistência temporal; bom desempenho apenas em simulação contínua "
        "pode ignorar a dinâmica de curto prazo. A consistência entre os dois "
        "indica representação adequada tanto da estrutura de longo prazo "
        "quanto da dinâmica de curto prazo."
    )


# ==========================================================================
#  Regionalização (§3.10) — Fase 2
# ==========================================================================
with tab_reg:
    st.markdown(
        """
        ### Regionalização para bacias não monitoradas

        A Fase 2 estende o modelo para aproximadamente 100 bacias
        brasileiras, avaliando a capacidade de generalização espacial
        necessária para a aplicação em bacias sem monitoramento
        fluviométrico (*Prediction in Ungauged Basins* — PUB).

        #### Três protocolos de validação espacial

        | Protocolo | Descrição |
        |:--|:--|
        | **R1 — Leave-One-Out** | Para cada bacia *i*, treinar com as demais e testar em *i*. Produz distribuição de NSE para *N* bacias. |
        | **R2 — *Split* espacial estratificado** | Divide as bacias em 70 % treino, 15 % validação e 15 % teste, estratificando por área, clima e uso do solo. |
        | **R3 — Transferência regional** | Treina em uma região (ex.: SP/MG) e testa em outra (ex.: PR/SC), avaliando generalização geográfica. |
        """
    )

    st.markdown(
        """
        #### *Encoder* de atributos

        Para aplicação em bacias não monitoradas, os parâmetros físicos
        são preditos a partir de atributos estáticos da bacia, por uma
        rede *feedforward* (MLP):
        """
    )

    st.latex(r"[\,t_{c\_scale},\ \sigma,\ \lambda\,] = \mathrm{MLP}(\mathbf{atributos})")

    st.markdown(
        """
        Atributos candidatos: área de drenagem, elevação média,
        declividade média, frações de uso do solo (MapBiomas),
        precipitação média anual e grupo hidrológico predominante.

        A estratégia opera em três etapas: (A) treinamento em bacias
        monitoradas para aprender os parâmetros físicos ótimos;
        (B) treinamento do MLP que mapeia atributos fisiográficos para
        esses parâmetros; e (C) aplicação em bacias não monitoradas
        utilizando apenas atributos e precipitação como entrada.
        """
    )
