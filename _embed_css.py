"""
CSS compartilhado pelos widgets — esconde sidebar, header e rodape do Streamlit
para que o embed no StoryMap apareca como se fosse um componente nativo.
"""
import streamlit as st


def hide_streamlit_chrome() -> None:
    """Esconde todos os elementos de UI do Streamlit exceto o conteudo."""
    st.markdown(
        """
        <style>
          /* Esconder sidebar inteira */
          section[data-testid="stSidebar"] { display: none !important; }
          div[data-testid="stSidebarNav"] { display: none !important; }
          button[data-testid="collapsedControl"] { display: none !important; }

          /* Esconder header e rodape */
          header[data-testid="stHeader"] { display: none !important; }
          footer { display: none !important; }
          #MainMenu { display: none !important; }

          /* Remover decoracoes do app */
          [data-testid="stDecoration"] { display: none !important; }
          [data-testid="stStatusWidget"] { display: none !important; }

          /* Ajustar padding para ocupar o iframe inteiro */
          div.block-container {
              padding-top: 0.5rem !important;
              padding-bottom: 0.5rem !important;
              padding-left: 1rem !important;
              padding-right: 1rem !important;
              max-width: 100% !important;
          }

          /* Remover scroll horizontal desnecessario */
          .main .block-container { overflow-x: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )
