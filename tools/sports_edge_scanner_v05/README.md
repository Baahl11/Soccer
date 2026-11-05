# SPORTS EDGE SCANNER v0.5 — Pro full-slate research + pregame refresh

**Estado:** Código construido y probado localmente el **4-nov-2026**; **NO desplegado, NO ejecutándose permanentemente, NO realizando apuestas**. Los datos Pro estaban disponibles mediante la conexión Soccer Edge API en esta conversación; el acceso HTTP directo requiere `API_FOOTBALL_KEY` configurada como secreto del ejecutor (no la compartas por chat). La renovación Pro mostró 7,500 consultas/día y datos de temporada 2026 en la prueba anterior, pero los saldos pueden variar.

## Qué incorpora v0.5

1. `pro-soccer-daily`: obtiene mediante la API directa la lista mundial de partidos para un día, comprueba integridad, selecciona las **ligas explícitamente configuradas** y ejecuta el SPORT FIRST para **TODOS** los partidos elegibles de esas ligas. El conector interno de ChatGPT devuelve hasta 100 y no sustituye la consulta completa.
2. `sports_edge.daily`: crea expedientes por partido (datos de la liga, estadísticas de ambos equipos); muestra `WATCH` para cualquier dato insuficiente. Clasifica perfiles descriptivos en dos shortlists **independientes**: señales de dominio y señales de goles. No son probabilidades ni ventajas. Córners queda **NOT VERIFIED** hasta modelarse con datos independientes.
3. Caché 90 min para estadísticas de equipo y 7 días para metadatos de cobertura; el presupuesto local de API-Football sigue siendo de **300 consultas/día**, configurable. Esto NO representa el saldo oficial y no incluye llamadas del conector de ChatGPT.
4. `pro-soccer-reference-odds`: consulta cuotas **únicamente después** de completar el screening deportivo de la jornada y guarda JSON de cada consulta con hora real de publicación; no inventa precio accionable de Caliente/Playdoit, ni modelos calibrados, ni EV.
5. `pro-soccer-refresh`: T-90 / T-40 / T-20 para un candidato deportivo válido: revisa que siga sin comenzar, consulta XI de ambos equipos, verifica si el proveedor identifica porteros y observa registros de lesiones. Si no hay lesiones reportadas, eso **NO** confirma una plantilla sana. La validación independiente oficial sigue siendo necesaria.
6. Plantilla de GitHub Actions **manual** en `.github/workflows/pro-soccer-research.yml`: corre solo si el código está en un repositorio GitHub y existe un secreto `API_FOOTBALL_KEY`. **No hay programación automática habilitada**. Incluye pruebas, auditoría y artefactos de investigación con vencimiento a 7 días.
7. **56 pruebas locales offline**, sin consumir crédito API ni simular picks reales.

## Un comando para una jornada

Se debe configurar **fuera del código y del chat** el secreto `API_FOOTBALL_KEY` en la máquina/runner que ejecute el scanner. Nunca lo copies a archivos `.env` versionados.

```bash
# Ejemplo de ligas conocidas del protocolo: 18 AFC Champions League Two y 17 AFC Champions League Elite
python -m sports_edge pro-soccer-daily \
  --date 2026-11-05 --leagues 18,17 \
  --output-dir data/2026-11-05 --odds-limit 5
```

Los archivos resultantes son:

- `raw_fixtures_YYYY-MM-DD.json`: respuesta completa original y auditoría del proveedor.
- `slate_YYYY-MM-DD.json`: lista elegible/excluida y control de fecha, estado, recuento y completitud.
- `sport_first_YYYY-MM-DD.json`: **todos** los partidos elegibles de las ligas elegidas, con errores `NOT VERIFIED` si faltan datos, shortlists de investigación y **cero** probabilidades inventadas.
- `reference_odds/*.json`: instantáneas de cuotas de casas realmente publicadas por el proveedor. La actualización de consulta **no cambia** la hora de la cuota en origen.
- `odds_research_YYYY-MM-DD.json`: estados y fallos de consultas de precios, solo WATCH.
- `daily_summary_YYYY-MM-DD.json`: resultado pre-final y bloqueos para BET.

### Pre-game para candidatos de la shortlist

```bash
# Usar el ID de un partido que esté en sport_first_...json, shortlist merged_fixture_ids
python -m sports_edge pro-soccer-refresh \
  --batch data/2026-11-05/sport_first_2026-11-05.json \
  --fixture-id 1630401 --output-dir data/2026-11-05 \
  --reference-odds
```

Si faltan alineaciones o el partido está fuera de ventana, informa `WATCH`. Un XI completo según el proveedor **NO** equivale a confirmación oficial independiente.

### Manual por etapas

```bash
python -m sports_edge pro-soccer-discovery \
  --date 2026-11-05 --leagues 18,17 \
  --raw-out data/day_raw.json --out data/day_slate.json

python -m sports_edge pro-soccer-batch-screen \
  --slate data/day_slate.json --out data/day_sport.json

python -m sports_edge pro-soccer-reference-odds \
  --batch data/day_sport.json --max-fixtures 5 \
  --output-dir data/day --out data/day_odds.json
```

### Prueba sin conexión

```bash
python -m unittest discover -s tests -v
```

En `reports/demo_v05_offline.json` hay una **demostración sintética etiquetada**, no una jornada real ni una predicción del modelo.

## Requisitos para activarlo en un servidor

- Un repositorio o servicio seleccionado por el usuario, con código v0.5 publicado en él.
- La clave API-Football Pro almacenada en el gestor de secretos del ejecutor; el archivo `.env.example` es una plantilla que **NO** contiene la clave.
- Elegir las competiciones elegibles del motor Soccer, por ID. `18,17` son solo ejemplos, no sustituyen la lista de competiciones aprobadas.
- Configurar un disparador (GitHub Actions, cron o servicio) y almacenamiento persistente de reportes/cuota/caché. La plantilla manual incluida **NO** lo activa por sí sola.
- Antes de cualquier BET real: modelo deportivo RAW independiente del precio, calibración, shrinkage, confirmación de XI y porteros, precio realmente accionable, cotejo de vig y límites de stake. **No están implementados ni conectados en v0.5**.

## Alcance y seguridad

Solo **SOCCER** está en el nuevo pipeline Pro. NFL/NCAAF y MLB tienen modelos separados, sin compartir pesos. No hay scraping ni automatización de Caliente/Playdoit, inicio de sesión a casas, apuestas ni pagos. No instalar dependencias externas adicionales. El módulo de combinaciones de la v0.4 se conserva, pero no puede emitir un boleto operativo sin los controles del Proyecto.

## v0.6 — read-only bridge to the existing Soccer engine (2026-11-05)

This is an **integration**, not a replacement/retraining of SOCCER EDGE ENGINE.
The GitHub paper-research job checks out the exact pinned `soccer-edge-mcp-v1`
commit (`108c193c50e59f12f19f8693fc05c5cc9c607db4`) under `engine/`
and exports `SOCCER_ENGINE_SOURCE_DIR` / `SOCCER_ENGINE_COMMIT_SHA`.
`sports_edge.engine_bridge.from_existing_engine` imports that checkout's
`mcp_gateway/soccer_model.py` and invokes `build_raw_projection` using only
verified same-fixture, same-league, same-season **sporting** provider inputs.
This occurs inside `sporting_research` **before** reference odds are consulted.
It adds a `raw_sport_projection` field to every screening row and
`raw_existing_engine_projections` to the JSON summary.

- Original upstream engine: `SOCCER EDGE ENGINE v1.0` /
  `GOAL_RATE_POISSON_LIMITED_v0.1` (these are the model's own version strings;
  no new weights are implied by this integration).
- Projection `MODELED_LIMITED`: pure Poisson-goals baseline, no verified
  xG/npxG/PPDA, no quantified XI impact, no out-of-sample calibration.
- A sample below five home/away league-split matches is marked explicitly.
- If the model source is absent, or API-Football IDs/seasons disagree, the
  bridge fails closed with `NOT_CONNECTED` / `NOT_MODELED`, not invented values.
- **Every projection remains WATCH** regardless of raw probabilities.
  No EV, stake or BET is promoted and no bookmaker is accessed by the model.
- The scheduled league allowlist now includes Conference League ID **848**.
- Historical state snapshots and data from earlier competitions are not used
  to form a new 2026 prediction. No bookmaker screenshot becomes training data.

### Tablero generado automáticamente
Cada ejecución válida produce `run_data/<fecha>/board_<fecha>.html` dentro
 del artefacto de GitHub Actions `soccer-paper-research`. Presenta todas las
probabilidades RAW `MODELED_LIMITED` de los partidos elegibles, destaca
muestras pequeñas y mantiene WATCH sin fabricar cuotas de apuestas.
Para abrirlo: Actions → ejecución del escáner → Artifacts → descargar
`soccer-paper-research` → abrir `board_<fecha>.html` localmente.

## v0.7 — Histórico multicompetición y mercado por mercado (2026-11-05)

La integración no se limita a H2H ni a las tres fechas de una copa:

- `sports_edge/history.py`: hasta 12 partidos oficiales **terminados** por equipo, de todas las competiciones accesibles, anteriores al kickoff del encuentro objetivo. Respuestas cacheadas, IDs cotejados. Si no se pueden verificar no se sustituyen con cero ni se afirma información de rivales inexistentes. Esos resultados se pasan al **mismo** `mcp_gateway/soccer_model.py` existente, sin momios; su lógica de mezcla sólo usa forma reciente cuando hay mínimo de datos.
- `sports_edge/market_context.py`: durante la etapa deportiva, para un máximo conservador de tres candidatos por ejecución, recupera estadísticas históricas de partidos reales: córners, faltas, tarjetas, tiros, tiros a puerta y formaciones *pasadas* con XI de once titulares. Promedios meramente descriptivos con `n` verificable, sin convertirlos en probabilidades de apuesta ni asumir la formación futura.
- `sports_edge/market_registry.py`: consulta de **solo lectura** los informes exactos de madurez por mercado de la rama original `soccer-edge-state`: 1X2, goles FT, BTTS, goles de equipo, 1H, 2H, córners FT/equipo, tarjetas, props, formación y faltas. Estados `RESEARCH_HOLD`, `REVIEW_ELIGIBLE` y `NOT_VERIFIED` se preservan, pero no promueven automáticamente ningún mercado a BET.
- `sports_edge/canonical_snapshot.py`: lee `/product/views` del motor desplegado en Render **sin llamar a `/internal/tick`** y exige timestamp reciente más `fixture_id` idéntico para mostrar filas del pipeline actual. Snapshots viejos quedan `STALE_OR_UNDATED_CANONICAL_PIPELINE`, sin predicciones atribuidas a la jornada.
- El tablero ahora incluye matriz de 12 familias, muestras de los últimos juegos, fuentes, razón de bloqueo y estadísticas descriptivas de candidatos. La hora de una consulta no sustituye el timestamp de un precio.

**Separación crítica:** un modelo de córners, faltas, mitades, tarjetas o emparejamientos de formación que existe en el repositorio **no significa que esté calibrado o promovido**. Los informes de validación de octubre de 2026 señalan límites OOS, CLV, y suficiencia muestral todavía pendientes. Para esos mercados, ninguna proyección ajena (p. ej. 2.5 goles FT) puede reutilizarse.

Configuración del flujo de GitHub Actions:

- `SOCCER_ENABLE_RECENT_HISTORY=1`
- `SOCCER_ENABLE_HISTORICAL_MARKETS=1`
- `SOCCER_ADVANCED_DEEP_DIVES=3` (máximo absoluto 6)
- `SOCCER_ENABLE_CANONICAL_VIEWS=1`
- `SOCCER_MARKET_EVIDENCE_DIR=<ruta de reportes originales de la rama soccer-edge-state>`

Las cuotas Caliente y Playdoit por selección y línea siguen siendo datos externos que deben verificarse con fuente, precio y momento de publicación. **Aún no hay EV ni recomendaciones BET.**