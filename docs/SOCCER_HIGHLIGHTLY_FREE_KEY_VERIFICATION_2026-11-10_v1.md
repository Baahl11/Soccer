# SOCCER EDGE — Highlightly BASIC key test and football coverage v1

**Fecha (México Central):** 2026-11-10, **as-of UTC:** 2026-11-11T04:56:09Z.  
**Resultado:** **API_KEY_ACCEPTED**. La clave del repositorio GitHub Actions \`SOCCER_HIGHLIGHTLY_API_KEY\` fue visible solamente para el job; **no se imprimió ni se expuso**.  
**Fuente oficial:** Highlightly Football API, \`https://soccer.highlightly.net\`, llamadas GET usando header \`x-rapidapi-key\`.
**Prueba auditada:** [GitHub Actions #42962412487](https://github.com/Baahl11/Soccer/actions/runs/42962412487); [artifact JSON #13829307773](https://github.com/Baahl11/Soccer/actions/runs/42962412487/artifacts/13829307773).  
**Workflow:** \`.github/workflows/soccer-highlightly-free-verify-v1.yml\` **manual-only** después de la ejecución, sin cron ni auto push. El workflow inicial tuvo error de sintaxis YAML y 0 jobs, corregido antes de ejecutar peticiones reales.  
**Estado de uso:** RESEARCH ONLY. Cero modificaciones de RAW/BET, cero consultas de odds, cero compras, cero escrituras de DB canónica.

## Observaciones medidas

| Consulta | HTTP | Filas devueltas | Paginación | Límite/remaining observado |
|---|---|---:|---|---|
| \`/matches?date=2026-11-09&limit=100\` | 200 | **88** | totalCount 88 | **100 / 99** |
| \`/matches?date=2026-11-10&limit=100\` | 200 | **70** | totalCount 70 | **100 / 98** |
| \`/matches?date=2026-11-11&limit=100\` | 200 | **100** | totalCount 149 | **100 / 97** |
| \`/matches/1403663161\` Premier League | 200 | un match devuelto | no aplica | 96 remaining |
| \`/statistics/1403663161\` | 200 | dos bloques de equipo | no aplica | 95 remaining |
| \`/lineups/1403663161\` | 200 | response sin conteo array esperado | no aplica | **94 remaining** |

Total **6 requests**, plan \`BASIC\`; 94 restantes **as-of** esa llamada (puede variar si otra aplicación consume la clave). La lista Nov11 incluye solo el primer page de 100/149: **no afirmar haber leído 149 partidos** ni 100% cobertura. Las fechas son filtros de UTC, no todos los fixtures elegibles por zona horaria México.

## Coincidencias con baseline API-Football

De la muestra no representativa de 12 fixtures canónicos observada antes por Soccer Edge API, se identificaron **6 coincidencias de nombres normalizados home+away** en los tres endpoints de calendario; IDs de referencia:

- 2026-11-09: 1570011, 1576529, 1600289.
- 2026-11-10: 1602548, 1619186, 1636031.
- 2026-11-11: 0 entre los primeros 100 registros.

**Advertencia crítica:** estos seis matches solo pasaron filtro de nombres **sin comparación de kickoff ni confirmación de liga ni IDs cruzados**. Son posibles candidatos para mapeo, NO seis identidades verificadas ni un 50% estadísticamente representativo. La paginación de Nov11 quedó incompleta. No realizar matching automático por nombre para migrar modelos.

## Datos que aparecieron realmente

El partido de muestra elegido de Premier League (\`matchId=1403663161\`) ofreció respuesta 200 en detalles, estadísticas y alineaciones.

- En \`/matches/{id}\`: familias con valores no nulos detectadas mediante nombres de campos: \`SHOTS\`, \`CORNERS\`, \`CARDS\`, \`POSSESSION\`.
- En \`/statistics/{id}\`: dos bloques con las mismas cuatro familias presentes.
- \`xG\`: **NOT VERIFIED** en la muestra; no se detectó un campo xG poblado.
- En \`/lineups/{id}\`: respuesta 200, pero el parser no confirmó XI/GK ni extrajo número de jugadores; **XI/GK = NOT VERIFIED**. Respuesta 200 no es prueba de once confirmada.

**Datos deportivos del partido observado son post-match o un snapshot actual, no necesariamente conocidos antes del saque inicial.** Nunca insertarlos retroactivamente en una proyección congelada ni afirmar capacidad pregame sin prueba as-of.

## Conclusiones prácticas y costes

1. El usuario subió la clave correctamente; la autenticación quedó probada en infraestructura GitHub Actions. **No hace falta pedirla otra vez ni enviarla por chat.**
2. **100 req/día** confirmadas por response headers; con 6 peticiones se obtuvieron 258 registros de calendario en 3 días, pero no todos los resultados/competencias se auditaron. Un día reportó 149 registros y solo se trajo un page.
3. Highlightly sí devolvió cuatro familias estadísticas reales en muestra; para certificar disponibilidad de xG, corners históricos por equipos, tarjetas por árbitro, player shot/saves y XI previo al match, se necesitan muestreos dirigidos.
4. La clase de producto \`BASIC\` no habilita odds según documentación del proveedor; **no** usar precio de referencia como betslip ejecutable.
5. Antes de compararlo contra API-Football en capacidad productiva, construir un corpus de **≥30 fixtures** por liga/temporada, fecha UTC, ambos IDs y timestamp, separando real observación pregame y postgame. Los proveedores tienen universos diferentes y no se puede estimar una cobertura porcentual a partir de este ejemplo.
6. Para no agotar cuota, reservar ≥70 requests/día en cada investigación, cachear fixtures por fecha/league, separar investigación del pipeline comercial y evitar polling de estadísticas en todos los encuentros.

## Próximos pasos de implementación propuestos

- **P1:** producir mapeo exacto y auditado de los 6 candidatos por fecha+liga+kickoff (con resolver alias previo), sin tocar IDs originales ni simular validación.
- **P2:** adaptador opt-in de datos Highlightly de *team stats/historical match stats* con provenance, as-of y cache. Límite inicial pequeño por tick, no encendido por defecto, sin precio/market en la fase SPORT.
- **P3:** mostrar en Edge Radar el resultado de cobertura real por familia con \`AVAILABLE/NOT VERIFIED/NOT MODELED\`, siempre respaldado por snapshots históricos y licencia para display.
- **P4:** solo tras mejora de datos con OOS y cotizaciones ejecutables se evalúa algún gate de mercado. **No BET** por el simple hecho de tener más datos API.

## Seguridad / citas

- [Highlightly Football API Docs](https://highlightly.net/football-api/documentation/)
- [Authentication](https://highlightly.net/sport-api/documentation/getting-started/authentication/)
- [Pagination, tiers and quota](https://highlightly.net/football-api/documentation/getting-started/pagination/)
- [Match statistics](https://highlightly.net/football-api/documentation/statistics/)
- [Benchmark anterior 13 llamadas públicas](SOCCER_PUBLIC_API_LIVE_BENCHMARK_2026-11-10_v1.md)
- [Plan de Edge Radar](SOCCER_EDGE_PRODUCT_OFFER_AND_FULL_SLATE_MARKET_READINESS_v1.0.md)

**Version:** \`SOCCER_HIGHLIGHTLY_FREE_KEY_PROBE_v1.0\`. Este reporte no altera versión del Soccer Edge Engine ni autoriza comercialización.
