# quali-storymap-embeds

Widgets Streamlit minimalistas para embed em iframe no StoryMap ArcGIS da
qualificação TTD-SCS-LSTM.

**Principal app (narrativa completa):** [quali-ttd-scs-lstm.streamlit.app](https://quali-ttd-scs-lstm.streamlit.app)
**StoryMap:** [storymaps.arcgis.com/stories/d3b585511b40427eb5fd1049a20b99c1](https://storymaps.arcgis.com/stories/d3b585511b40427eb5fd1049a20b99c1)

## Por que um app separado?

Os embeds do StoryMap aparecem dentro de iframes e são lidos em contexto
narrativo. O app de apresentação completo tem sidebar, cabeçalho e várias
páginas — tudo isso polui o iframe. Aqui cada página é um componente único,
com sidebar e header escondidos via CSS, dimensionado para caber limpo dentro
de um bloco *Embed* ou *Sidecar* do StoryMap.

## Widgets

| Página | URL | Usado na seção |
|---|---|---|
| `1_mapa_telemetricas.py` | `/mapa_telemetricas` | "A rede telemétrica" |
| `2_tradeoff.py` | `/tradeoff` | "O trade-off entre previsão e simulação" |

## Rodando localmente

```powershell
cd D:\TTD_SCS_LSTM\quali\storymap_embeds
uv run streamlit run app.py
```

## Deploy

Streamlit Cloud, vinculado ao repo `viniciusazeved/quali-storymap-embeds`.
Cada push no `main` dispara redeploy automático.

## Dados

Os widgets leem de `data/`:
- `estacoes_telemetricas_{ana,todas}.shp` — inventário HidroWeb/ANA (abril/2026)
- `summary.csv` — métricas NSE por modelo × horizonte (ablação Fase 1)
- `summary_continuous.json` — métricas NSE da simulação contínua

Para regenerar os shapefiles: rodar
`D:\TTD_SCS_LSTM\quali\Apresentacao\shp\_export_shapes.py`.
