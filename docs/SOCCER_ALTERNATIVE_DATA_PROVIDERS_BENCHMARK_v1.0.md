# SOCCER EDGE — APIs alternativas, costos y cobertura realmente utilizable v1.0

**Consulta de tarifas y documentación pública:** 2026-11-10  
**Estado:** investigación de proveedores / NO autorización de compra, migración de modelo, nuevos precios de apuestas o publicación de datos de terceros.  
**Regla:** SPORT FIRST / MARKET SECOND. Datos sin identidad, hora y permiso de uso verificados = NOT VERIFIED. Cada fuente debe seguir aislada hasta pruebas.

## Decisión recomendada

1. **Mantener API-Football como fixture registry y fuente primaria** durante la transición. Es la única con IDs/ingesta y datos históricos ligados al motor actual. Cambiar en caliente rompería hashes, as-of y OOS; nunca crear una correspondencia de fixtures sin verificar equipo+liga+temporada+kickoff+IDs.
2. **Primero optimizar cuota y caché, no reemplazar la API.** API-Football anuncia Pro **US$19/mes, 7,500 req/día** y Ultra **US$29/mes, 75,000 req/día**. La última investigación interna se quedó en **35/40 requests** por guardrail local de canary, no por evidencia de agotamiento del límite diario pagado. Subir el plan sin rediseñar ese guardrail no aumentará automáticamente el número de modelos.
3. **Piloto OpenFootAPI para xG real donde exista, sin usarlo como sustituto universal.** Developer **US$14/mes**, 250,000 req/mes y xG/shot maps en Premier League, La Liga, Bundesliga, Serie A y Ligue 1; catálogo general ~120 competiciones; XI/eventos de algunas competiciones. Gratis para fixtures/standings con 5,000 req/mes, **pero la funcionalidad de xG requiere Developer**. Fuente xG de competiciones no soportadas = NOT VERIFIED.
4. **Highlightly Pro para evaluar cobertura básica más barata:** **US$9.49/mes**, 7,500 req/día, 950+ ligas anunciadas, estadísticas, corners, tarjetas, alineaciones y jugadores. **No se ha verificado feed de xG avanzado ni la granularidad/licencia para cada liga**; puede sustituir parte del registro si supera benchmarking, no reemplazar a ciegas el modelo.
5. **FootyStats si la prioridad son córners/tarjetas, no por precio:** Hobby **£29.99/mes**, hasta 50 ligas (la versión ES del sitio muestra 40; confirmar el contrato), 1,800 req/hora, métricas de xG/tiros/córners/tarjetas y BTTS/O-U por equipo/partido; **no tiene datos en vivo** según su FAQ. Útil para investigación estadística histórica de determinados campeonatos. No es más barato que US$19/mes.
6. **Sportmonks para un piloto de máxima profundidad en cinco ligas:** Starter **€29/mes**; un endpoint con includes de fixture/stats/lineups; disponibilidad exacta de xG/odds debe validarse contra el plan/add-ons contratados. No cubre todas las ligas en Starter.

### Tabla de decisión (precios publicados, NO contrato vigente del usuario)

| Proveedor | Precio mensual público | Cuota / ligas | xG y corners/players | Mejor uso para Soccer Edge | Bloqueo |
|---|---|---|---|---|---|
| API-Football | Free $0; Pro **$19**; Ultra **$29** | 100 / 7,500 / 75,000 req diarios; todas las competiciones declaradas | Estadísticas por fixture/players, lineup e injury variable. La capa RAW no tiene xG confiable | Canonical fixture ID, calendario, estadísticas actuales, odds proxy separadas | No precio betslip ejecutable ni xG/PPDA universal |
| Highlightly | **$9.49** Pro | 7,500 req/día, 950+ ligas anunciadas | Córners, tarjetas, SOT, jugadores, XI anunciados; xG avanzado **NOT VERIFIED** | Prueba barata de amplitud y frescura | Cobertura real por competición, licencias, match IDs |
| OpenFootAPI | Free $0; **$14** Developer; $39 Pro | 5,000/mo free; 250,000/mo Developer; catálogo ~120 | xG + shot maps solo cinco grandes ligas europeas, XI/eventos limitados por liga | Enriquecimiento xG de selección restringida | No sustituye calendar/markets universales; fuente/atribución por competición |
| FootyStats | **£29.99** Hobby | 50 ligas EN / 40 ES, 1,800/h; sin live | xG, shots/SOT, corners, cards, BTTS, equipos/jugadores según cobertura | Datos históricos para modelos de córners, cards y primer análisis | Más caro; mismatch de paquetes en páginas oficiales; sin live |
| Sportmonks | **€29** Starter | 5 ligas y 2,000 requests por entidad/hora | Incluye datos profundos con includes; confirmar xG/odds por add-on contratado | Piloto de pocos torneos y profundidad | No todos los fixtures de Soccer, licencia/price tier por liga |
| football-data.org | €0; €12/mes live; €29/mes deep | 12 competiciones; rate-limit por minuto | Lineups/cards en paquete Deep; xG no confirmado | Backup barato de calendario en ligas europeas | Cobertura mucho menor, xG no probado |

**No comparar como si $/mes, €/mes y £/mes fuesen iguales; no se han calculado FX.** Importes sujetos a cambios, impuestos, región, uso y licencia.

### Fuentes oficiales revisadas

- API-Football pricing: https://www.api-football.com/pricing
- API-Football v3 docs: https://www.api-football.com/documentation-v3
- Highlightly fútbol y precios: https://highlightly.net/football-api/
- OpenFootAPI pricing: https://openfootapi.com/pricing
- OpenFootAPI coverage: https://openfootapi.com/coverage
- OpenFootAPI docs: https://openfootapi.com/docs
- OpenFootAPI terms: https://openfootapi.com/terms
- FootyStats API: https://footystats.org/api
- FootyStats FAQ: https://footystats.org/es/api/documentations/faq
- Sportmonks pricing: https://www.sportmonks.com/football-api/plans-pricing/
- football-data.org: https://www.football-data.org/pricing

## Importante: derechos y procedencia

- OpenFootAPI permite retener/agregar información para investigación interna, **pero no otorga automáticamente derecho a redistribuir datos de terceros**. Cada respuesta lleva \`meta.sources\`; las licencias de origen, atribución y restricciones pueden diferir por competición. No copiar mapas de tiro ni raw feed al público sin permiso. Ver https://openfootapi.com/terms.
- FootyStats anuncia uso en sitios/apps, retención e investigación, **pero prohíbe revender y redistribuir raw feed**. Ver FAQ oficial.
- API-Football/Highlightly/Sportmonks: confirmar con proveedor contrato de cliente, uso derivado, logos, caching, derechos de publicación, odds, territorio y respaldo. No confiar solo en etiquetas de un comparador.
- Sin nueva key autenticada ni permiso de compra, **no se han hecho consultas de producción a ninguna API candidata**; cifras de cobertura declaradas por vendedor **NO** son benchmarking contra nuestro slate.

## Prueba A/B en paralelo, definida antes de pagar

### Paso 0 — sin costo ni keys de terceros

Extraer de nuestro fixture registry una muestra reproducible de al menos 30 partidos de:
- Liga MX y otras ligas LATAM;
- las cinco grandes ligas europeas;
- ligas con historial de córners / tarjetas;
- competiciones donde API-Football tiene huecos documentados.

No usar cuotas para elegir muestra. No hacer apuestas de investigación basándose en comparación de precios.

### Paso 1 — identity mapping

Guardado privado \`provider_fixture_map\` sólo para los fixtures probados, con IDs por proveedor y:
\`league, season, country, kickoff_utc, home_id/name, away_id/name, match_status, observed_at_utc, provider_name, source_rights\`.
Auto-match únicamente con evidencia inequívoca (nunca empatar a nombres parecidos solamente). Si ambiguo => \`IDENTITY_NOT_VERIFIED\`.

### Paso 2 — obtener y medir lo que realmente entrega el proveedor

Para cada \`fixture_id\`, mercado y etapa pregame:
- \`fixture_coverage\`: ¿existe el partido correcto?
- \`team_stats_coverage\`: historial before kickoff verificable, n de muestra, splits casa/fuera.
- \`xg_coverage\`: xG verdadero con source timestamp, competencia, tipo xG/npxG. No confundir modelo derivado "goal expectancy" con xG por tiros medidos.
- \`corners/cards/shots/players\`: partidos históricos con datos por equipo y jugador y fecha en que fueron publicados.
- \`lineup/availability\`: true XI/GK y lesión/sanción material con fuente oficial y hora.
- \`latency/quota/cost\`: respuestas vacías, errores, llamadas por fixture, latencia, cobertura fresca a T-90/40/20.
- \`rights\`: posibilidad contractual de persistir internamente y mostrar agregados al cliente.

Un 404 o un campo null **cuentan como incompletos**, nunca como cero ni como un partido sin córners.

### Paso 3 — separar datos de modelos

Si el proveedor mejora cobertura, integrar un adapter opt-in **deshabilitado por defecto** que:
- no cambie el \`fixture_id\` canónico;
- conserve cada \`provider, observed_at, source_updated_at, license, sample_n, raw_hash\`;
- no use fair odds/modelos del proveedor para generar nuestra proyección RAW;
- no reescriba pronósticos congelados/históricos ni aprenda de postmatch sin entrenamiento as-of;
- no trate un player match stat *pospartido* como un dato disponible antes;
- no aumente automáticamente la Availability Confidence;
- no consulte mercado hasta después de la proyección deportiva;
- rechace nuevos datos para uso de subscriber si licencias no están revisadas;
- respete budget fijo por ejecución y quota headers.

### Gate de adopción

**Elegir un nuevo proveedor solo si** demuestra sobre muestras pareadas mejor cobertura **de la familia prioritaria**, fuente pregame confiable, licencia/costo compatibles, estabilidad y regresión de modelo satisfactoria. Una web que dice "950 leagues" no prueba 950 ligas aptas para un modelo de córners.

**Siguiente adquisición mínima que recomendaría:** prueba gratuita de Highlightly para comprobar amplitud, y *solo después* un piloto temporal OpenFootAPI Developer de xG sobre ligas top-five si derechos y necesidad están confirmados. No contratar ambos antes del benchmark. Alternativa: si el objetivo urgente es abaratar solicitudes, optimizar cache bajo el API-Football ya contratado, sin gastar.

## Impacto en Edge Radar

El PR runtime #153 añade contabilidad visible de \`RAW_SPORT_RECORDED / PARTIAL_ANALYSIS / SPORT_EVIDENCE_ONLY / MARKET_ONLY / NO_PERSISTED_ANALYSIS\`, fuentes y motivos faltantes desde registros de datos ya existentes; **no consume peticiones nuevas ni implica que se realizó un sport screen íntegro**. Esto permite saber qué tan grande es la brecha antes de habilitar otro proveedor.

**No hacer ahora:** compra, cambio de claves, aumentar cuota 40 a ciegas, cambiar model weights, publicar xG no verificados o comercializar mercados no promovidos.

**Siguiente éxito medible:** 100% de fixtures elegibles del registry contabilizados en Radar; demostrar con solicitudes reales cuántos pueden modelarse y cuántos quedan sin datos. Solo entonces comparar nuevos proveedores sobre fixtures previamente elegidos.

**Versión:** DATA_PROVIDER_BENCHMARK_v1.0 — documentación no cambia SOCCER EDGE ENGINE v1.0 ni las familias autorizadas para apuestas.
