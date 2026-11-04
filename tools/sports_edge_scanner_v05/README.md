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