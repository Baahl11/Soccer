# SOCCER EDGE — Propuesta de suscripción y readiness de análisis integral v1.0

**Fecha de evaluación:** 2026-11-10 (America/Mexico_City)  
**Proyecto:** SPORTS EDGE ENGINE / Soccer  
**Tipo:** estrategia de producto, diagnóstico técnico y plan de aceptación  
**Estado:** PROPUESTA DOCUMENTADA — beta de investigación condicionada; **BET comercial / cobro público = NO-GO**  
**Mandato invariable:** **SPORT FIRST. MARKET SECOND.** No falsear cobertura, precios, XI, lesiones, resultados, rendimiento ni disponibilidad.

> **Decisión de producto propuesta:** ser una plataforma de inteligencia de fútbol que priorice la información verificable, el ahorro de tiempo, la personalización y la transparencia; no vender un volumen fijo de picks ni promesas de ganancias. Esta propuesta no modifica pesos, thresholds ni autorizaciones de mercados del SOCCER EDGE ENGINE MASTER v1.0.

## 1. Resumen ejecutivo

La oportunidad comercial está en cuatro funciones vinculadas por la rutina del usuario:

1. **Edge Radar:** analizar el calendario completo a nivel *screening*, identificar partidos para investigación y dejar explícito qué quedó sin analizar y por qué.
2. **Match Intelligence:** explicar el contexto deportivo, las proyecciones RAW y el nivel de verificación, por mercado y con su tamaño de muestra. Solo después de la proyección deportiva, y cuando haya mercado auténtico, mostrar comparación contra cuota.
3. **My Edge:** preferencias por liga/equipo/mercado, partidos guardados y avisos de cambios relevantes con caducidad, consentimiento y control de frecuencia.
4. **Performance & Trust:** registro inmutable de pronósticos originales, resolución oficial FT, error y calibración, con investigación/OOS completamente separada de BET liquidados y CLV verdadero.

**Promesa al cliente, cuando podamos soportarla:** "Menos ruido. Más análisis. Decisiones con evidencia".  
**Prohibido:** «87% accuracy», «ganancias garantizadas», «picks seguros», «todas las ligas 100% modeladas», «cuota ejecutable de Bet365» sin fuente directa, o contar estudios como BET rentable.

**Estrategia:** primero beta privada gratuita de investigación con permisos de datos, privacidad y accesos revisados; luego monetización PRO únicamente tras aprobación legal, seguridad, billing, derechos de visualización y, para picks/alertas BET, gates de predicción *por familia*.

## 2. Qué recibe el usuario

| Módulo | FREE (propuesta) | PRO (objetivo condicionado) | Política |
|---|---|---|---|
| Calendario / Today | Partidos, horarios, status, cobertura básica, WATCH/PASS verificados | Radar completo y filtros avanzados | La existencia del fixture no implica análisis completo |
| Edge Radar | Selección limitada de partidos para seguir e introducción al método | Ordenamiento y explicación deportiva por ligas y mercados, motivos de inclusión/exclusión | No inventar scores ni completar faltantes |
| Match Intelligence | Datos descriptivos permitidos y explicación SPORT FIRST | Profundidad de evidencia, distribución de goles, disponibilidad y model status, trazabilidad de fuente | Mostrar NOT VERIFIED / NOT MODELED donde aplique |
| Comparación de mercado | Glosario y ejemplo educativo | Precios exactos actuales + probabilidades RAW/SHRUNK/FAIR/EV **solo si verificados** | Nunca convertir cuotas proxy/API en betslip ejecutable |
| My Edge | Historial personal básico cuando Auth/RLS esté validado | Guardados y preferencias de liga/equipo/mercado, seguimiento de estados | Persistencia con permisos por usuario |
| Alertas | Resumen opcional de seguimiento si existe consentimiento | Avisos de XI, disponibilidad, invalidación, precio; alertas BET solo para mercados autorizados | Push está apagado; TTL, dedupe, opt-in y silencio |
| Performance | Metodología, conteos transparentes y límites de investigación | Histórico detallado y vistas de calibración **donde la licencia y el acceso lo permitan** | Nunca mezclar investigación con ROI de apuestas reales |
| Picks/Leans | Sin BET inventados; estados solo emitidos por motor | BET/LEAN solo si la correspondiente familia supera gates de evidencia | 0 BET es respuesta válida |

**Valor que sostiene la renovación:** disminuir trabajo manual, explicar decisiones, refrescar datos cuando realmente cambian, personalizar seguimiento y rendir cuentas públicamente. La suscripción no compra certeza ni una cantidad fija de apuestas.

### Posicionamiento y público inicial

- Público: aficionados analíticos de fútbol de habla hispana que valoran estadísticas, filtros, alertas verificables y transparencia. Validar intereses reales por entrevistas, no asumir tracción.
- Diferencia frente a tipsters: trazabilidad, muestra y timestamp, razones de WATCH/PASS y modelo reproducible.
- Diferencia frente a dashboards genéricos: flujo integrado SPORT → disponibilidad → mercado → decisión → resultado.
- **NO** comercializarse como una casa de apuestas, servicio de ejecución o predicción infalible.

### Pricing: hipótesis, no lanzamiento

La hoja histórica de frontend documenta Stripe configurado como **MX$249/mes** para MX_LATAM y **US$14.99/mes** para US. No es aprobación de precio actual, ni confirmación vigente de cobro/refund, ni permiso para activar billing. Probar disposición a pagar por utilidad y renovación, no por ROI prometido.

El interruptor de cobro público \`SOCCER_PUBLIC_BILLING_ENABLED\` debe continuar **false** hasta pruebas sandbox, obligaciones de usuario, legales y aprobación expresa. No efectuar cargos en un piloto de investigación.

## 3. Diagnóstico verificado — no confundir capas

**Fecha y fuentes de observación:** auditoría del repo \`soccer_edge_state/analysis/soccer_commercialization_readiness_v1.json\` de **2026-11-10T22:34:01Z**; acciones reales [#42891660125](https://github.com/Baahl11/Soccer/actions/runs/42891660125); código \`mcp_gateway/automation_v5.py\`, \`mcp_gateway/soccer_model.py\`, \`.github/scripts/soccer_isolated_sport_only_v1.py\`; planes de [lanzamiento](SOCCER_COMMERCIALIZATION_LAUNCH_GATES_v1.md) y [frontend](../17_SOCCER_FRONTEND_PRODUCT_ROADMAP_v1.0.md). Los contadores son *as-of*, no promesas sobre futuras jornadas.

| Capa | Evidencia actual | Diagnóstico |
|---|---|---|
| Descubrir calendario | Ejecución prospectiva enumeró **316** fixtures del rango de 32 horas | **DEMOSTRADO para listado**; no significa 316 modelos |
| Detectar cobertura | **14** candidatos Tier A/B con ventana válida en la ejecución; ligas C/D no candidatas BET | **PARCIAL**; cobertura varía por competición/temporada/partido |
| Procesar todos los elegibles | **8 de 14** candidatos recibieron RAW limitado; **6** quedaron pendientes por presupuesto; cap **35/40** llamadas utilizadas | **NO COMPLETO**; admitir explícitamente el backlog |
| Modelo deportivo básico | \`soccer_model.py\` usa promedios de goles home/away + forma reciente cuando está; Poisson para 1X2/BTTS/totales FT | **RESEARCH / MODELED_LIMITED**; sin xG/PPDA/field tilt demostrados en esta capa |
| Triple sporting screen | Código calcula \`side_edge_score\`, \`goal_environment_score\`, \`two_way_scoring_score\`; \`corners_opportunity_score = NOT MODELED\` | **INCOMPLETO respecto al Master**; no son tres tracks plenamente independientes |
| Modelos RAW persistidos | **8** proyecciones refinadas del canary fueron aceptadas por endpoint privado OIDC (8 nuevas filas según ACK, 1 POST) | **DEMOSTRADO por ACK**; no convertir en pronósticos BET ni en backtest |
| Investigación baseline histórica | Auditoría reportó **13** modelos prepartido con commit previo al inicio y **7** partidos FT oficiales unidos a OOS | **Muestra insuficiente**; conjunto distinto, no sumar automáticamente con las 8 refinadas |
| Quotes y precios | **16** visitas de precio proxy API-Football; **0** pares exactos con avance independiente de \`provider_updated_at\`, **0** mercado directo ejecutable verificado | **BLOQUEADO** para precios BET |
| Modelo/mercado validado para producción | **7** familias RESEARCH, **3** SHADOW y **0** promovidas a BET; CLV histórico canónico **789** (no de este piloto) | **NO-GO**; CLV agregado no valida mercados sin settlement específico |
| Frontend y cuentas | FE-0–FE-6/FE-8 preview, FE-7 Auth/Stripe guardado, FE-9 PWA instalable; push apagado | **Base construida, QA comercial pendiente** |
| Acceso al nuevo snapshot refinado desde app | Existe **POST privado de ingestión** \`/internal/isolated-sport-refined-pg-v1/import\`; no se ha verificado endpoint customer-read con RLS para esta nueva tabla | **FALTA PUENTE DE LECTURA CONTROLADA** |

**Cuidado con los denominadores:** los 316 fixtures incluyen encuentros de distinta calidad/estado; "14" es el subconjunto candidato A/B según ventana de la corrida; "8" son RAW limitados, **no deep dives completos**, y "0 mercados productivos" es el dato decisivo para BET. No atribuir a todo el sistema un porcentaje global inventado.

## 4. Cobertura por familia de mercado: implementado ≠ modelado ≠ validado

Estado procedente de auditoría 2026-11-10 22:34Z y revisión del modelo base. "Código de investigación" y "SHADOW" **no** son un producto que pueda ofrecer BET.

| Familia de mercado | Capacidad deportiva automática actual | Evidencia de madurez de mercado | Brecha imprescindible |
|---|---|---|---|
| **1X2** | Poisson básico RAW Home/Draw/Away, sí; no xG avanzado ni impacto XI cuantificado | **SHADOW**; 53 CLV verdaderos históricos, 10 BET liquidados en auditoría; no autorizado | Profundidad, discriminación de empate, OOS/ROI/availability y quote ejecutable |
| **Goles FT O/U** | RAW over 1.5/2.5/3.5 y distribución, sí pero **limitado** | **RESEARCH**; 8 CLV verdaderos, 15 liquidados; sin promoción | Ajuste rival/xG, calidad de fuente, mercado exacto y mayor muestra |
| **BTTS** | RAW Yes/No de Poisson básico, sí | **SHADOW**; 20 CLV verdaderos y **0** liquidados en auditoría; no autorizado | OOS/settlement, disponibilidad, precio independiente |
| **Team totals** | Datos de goleadores/por equipo sirven como inputs; todavía sin modelo independiente de mercado completo acreditado | **SHADOW**; 708 CLV históricos de dos lados juntos, **0** liquidados; no autorizado | Modelo por equipo/threshold con reglas de settlement y validación real |
| **DNB / Double Chance / Asian Handicap** | Puede investigarse desde distribución de marcadores, pero el parser del \`soccer_model.py\` actual **no los evalúa** como mercados; push/line/margen no completados | **No evidencia de promoción independiente** | Mapear líneas/selecciones y push, probar prob. de cobertura, vig y settlement |
| **1H y 2H** | No hay proyección autónoma aprobada por mitad en la ruta base | **RESEARCH**, CLV verdadero **0** en ambas familias | Datos por periodo + modelos separados con calibración y reglas de selección |
| **FT Corners** | \`corners_opportunity_score = NOT MODELED\` en ruta RAW automática; existen herramientas de perfil histórico/investigación separadas | **RESEARCH**, CLV verdadero **0** | Adquirir históricos verificables y modelo de córners por fixture/liga/forma |
| **Team Corners** | Sin proyección deportiva automática propia y completa | **RESEARCH**, CLV verdadero **0** | Distribución por equipo, localía, rival y as-of, OOS por lado |
| **Cards / árbitro** | Investigaciones separadas; no hay completo modelo validado de mercados de tarjetas en RAW básico | **RESEARCH**, CLV verdadero **0** | Árbitro/estilo/disciplina, yellow/red separados, historial as-of y OOS |
| **Player Props** (shots, SOT, goles, asistencias, tarjetas, GK saves) | Investigación y módulos puntuales; no conjunto completo validado con minutos/XI y línea exacta | **RESEARCH**, CLV verdadero **0** | Confirmación rol/minutos, cobertura histórica real, OOS por prop, precio directo |
| **Bet Builder / SGP** | Especificación y perfiles descriptivos; falta distribución conjunta calibrada | Ticket conjunto ejecutable = **NOT VERIFIED** | Modelar correlación, fuente y cuota de ticket real, settlement completo |

\`soccer_model.py::_market_family\` reconoce actualmente **1X2, BTTS y TOTAL FT**; esta enumeración **no** implica que otros módulos aislados no existan, sino que la vía automatizada base no hace un deep dive completo sobre todos ellos.

**Secuencia de mercados propuesta:** validar primero 1X2 + FT goals + BTTS en su propia familia; después team totals y DNB/AH; después corners/team corners; luego 1H/2H, cards y props donde exista cobertura; por último builders/SGP. Las prioridades comerciales no permiten adelantar familias que no superan OOS.

## 5. Qué aporta API-Football v3 y qué NO resuelve por sí sola

La [documentación de API-Football](https://www.api-football.com/documentation-v3) describe:
- \`/fixtures\`, \`/leagues\`, \`/teams/statistics\`: calendario, metadatos, coverage de la liga y estadísticas acumuladas.
- \`/fixtures/statistics\`, \`/fixtures/players\`: corners, tiros, SOT, tarjetas, salvadas y acciones de jugador cuando el partido/temporada esté cubierto.
- \`/fixtures/lineups\`, \`/injuries\`: XI, formación y reportes de ausencias cuando existan, con disponibilidad a menudo muy cerca del inicio o posterior.
- \`/players\`, \`/players/squads\`: perfiles, agregados y plantillas; exigir paginación y timestamp de disponibilidad.
- \`/odds\`: cuotas de referencia **agregadas por proveedor**, no demostración de precio ejecutable directo de un sportsbook; actualizaciones pregame relativamente espaciadas y cobertura variable.

**NO asumir** que provee un flujo completo y uniforme de xG/npxG/PSxG, big chances, PPDA, field tilt, sanciones oficiales actuales, clima y precios ejecutables. La salida de la API puede tener coverage=true y sin observación real; **ausente no es cero**. Integrar proveedores *licenciados* adicionales por familia cuando esos datos sean materialmente necesarios, o etiquetar NOT VERIFIED / NOT MODELED.

**Licenciamiento:** las [condiciones de API-Football](https://www.api-football.com/terms) prohíben reventa directa de datos sin permiso. Un producto de análisis derivado, visualización de datos, logos y derechos de distribución necesita revisión de términos/permiso; no basta que una petición API devuelva un JSON.

## 6. Arquitectura objetivo: todos los partidos se auditan, no todos consumen deep dive

La definición correcta de "analizar todos los matches" debe distinguir **inventario de calendario para 100% de los elegibles** de **modelo completo de todos los mercados para cada partido**. El segundo objetivo no es honesto si faltan inputs, cobertura y cuota.

\`\`\`text
API + fuentes oficiales
  -> 0. Calendario completo / estados / competencia / timezone
  -> 1. Cobertura y calidad por partido y familia (A/B/C/D)
  -> 2. Screening deportivo barato para TODOS los elegibles con datos
       (SIDE / GOALS-BTTS / CORNERS y luego familias opcionales)
       | SIN DATOS -> NOT MODELED + reason
  -> 3. Shortlist deduplicada y prioridad, sin líneas de apuestas
  -> 4. Deep dive por mercado/partido donde los datos lo soporten
       (forma/rival, XI/GK/lesiones, históricos por periodo,
        tiros/SOT/corners/cards, muestras reales, inputs as-of)
  -> 5. Guardar RAW inmutable, versión + hashes + timestamps
  -> 6. Consultar SOLO mercados de shortlists ya modeladas
       (bookmaker/selección/línea/cuota/provider_update/timestamp)
  -> 7. Fair devig -> shrink -> edge/EV -> BET/LEAN/WATCH/PASS
  -> 8. Invalidate/recheck T-90/T-40/T-20 / recordatorio
  -> 9. Official FT -> OOS/calibration/settlement/CLV (separados)
  -> 10. API cliente con derechos de datos, Auth/RLS y redacción FREE/PRO
\`\`\`

**Regla de cobertura de ingeniería:** para cada fixture elegible conservar uno de \`SCREENED_VALID\`, \`NOT_MODELED_DATA_MISSING\`, \`DEFERRED_API_BUDGET\`, \`DEFERRED_TIME\`, \`EXCLUDED_STATUS\`, \`NOT_ELIGIBLE_DATA_TIER\`. No confundir \`fixture_scan_count\` con \`screened_sport_count\`. Reportar por familia \`candidate_count\`, \`model_attempt_count\`, \`raw_valid_count\`, \`full_deep_dive_count\`, \`market_quote_count\` y \`bet_allowed_count\`.

### Restricciones reales de capacidad

El canary refinado limita **40 requests salientes por ejecución**, **20 candidatos** en cola y **32h** máximo opcional. En una corrida real, **35** requests alcanzaron **8** modelos, de **14** A/B. El v5 canónico tiene por defecto **8 deep dives por tick**, sujeto a cuota/caché/stages. No se debe afirmar "cubre todos los mercados de los 316 juegos" bajo ese presupuesto.

**Ajustes propuestos antes de ampliar el alcance:**
1. cache persistente y compartida para \`/leagues\`, team stats y fixtures anteriores sin repetir datos estáticos;
2. separar presupuesto de *descubrimiento* de presupuesto de *screen/deep-dive/refresh* y métricas por fase;
3. consultas agrupadas cuando API permita \`ids\`, deduplicación por fixture y refresh selectivo;
4. cola que sobreviva a ticks con contadores \`deferred_due_to_budget\` y prioridad deportiva, nunca basada en cuota;
5. cobertura estadística *por mercado* y ligas configurables, rechazar datos carentes de source-as-of;
6. correr pruebas de costo/capacidad con cache fría y caliente antes de subir cuota, memoria o frecuencia;
7. verificar que un partido con córners sin datos no se clasifica como un partido de cero córners.

## 7. Backlog secuencial y gates medibles

### P0 — Diagnóstico y cobertura de slate

- [ ] Métrica de universo completo: \`fixtures_discovered\`, \`eligible\`, \`screened\`, \`not_modeled\`, \`deferred\`; suma reconciliada por fixture ID y estado.
- [ ] Daily discovery para ligas objetivo + próxima fecha y pregame refresh con cuota protegida, sin omisiones silenciosas.
- [ ] Coverage por mercado/temporada/fixture; registrar estadística vacía y motivo, no inferir salud.
- [ ] Caché y colas para procesar todos los fixtures elegibles en *screen barato* o clasificarlos en una razón explícita; nunca elevar límites a ciegas.
- **Gate P0:** corrida con slate completo y reporte de **cada fixture elegible**, incluido todo lo omitido, sin exceder presupuesto. P0 **no** exige full deep dive de un partido sin datos.

### P1 — Tres modelos deportivos independientes confiables

- [ ] Side, Goals/BTTS, Corners implementados realmente como tres screens independientes; el tercero no puede quedar \`NOT MODELED\` por defecto si se ofrece públicamente.
- [ ] Guardar RAW con historial que solo use información disponible antes de kickoff, versiones, fingerprints y diagnóstico de cobertura.
- [ ] Resolver disponibilidad: XI/GK/lesiones verificadas con fuente y timestamp; impacto no cuantificado permanece UNKNOWN y reduce confianza.
- [ ] Validar por familia benchmarks, Brier, log loss, sesgos por liga y error en partidos out-of-sample **sin rehacer forecasts**.
- **Gate P1:** mostrar \`MODELED_LIMITED\` cuando ese sea el modelo; publicar probabilidades de familia como "experimentales" hasta que su evidencia de calibración lo permita.

### P2 — Mercados que existen de verdad y decisiones por familia

- [ ] Market adapters exactos de 1X2, FT Totals, BTTS con precio-book-timestamp y proveedor actualizado **después del RAW**.
- [ ] Separar DNB/AH/DC, team totals, 1H/2H, corners, cards y props en módulos y suites independientes; no reutilizar precios ni modelos sin prueba.
- [ ] Probar devig, settlement, push, line mapping, caducidad y actualización, disponibilidad; pasar por RESEARCH -> SHADOW -> BET solo con OOS por familia.
- [ ] Diferenciar quote proxy de precio ejecutable; bloquear apuestas cuando falte ejecución directa y control de identidad.
- **Gate P2:** cada mercado soportado publica su \`data_tier\`, \`raw_status\`, \`quote_status\`, \`oos_status\`, \`decision\`. Familia sin quote exacta = WATCH máximo.

### P3 — Producto beta visible con seguridad y derechos

- [ ] API de lectura autenticada y limitada para los **8 refined RAW** de investigación, con datos derivados autorizados para mostrar. POST OIDC ya existe; lectura customer nueva **no verificada**.
- [ ] Conectar radar y Match Center del preview; NO publicar features crudas ni inventar logos o precios.
- [ ] QA mobile/desktop y estados FREE/PRO; verificación de Supabase RLS, age/geo/privacy, permisos de licencia.
- [ ] Guardados; notificaciones in-app solo con consentimiento y preferencias; push no habilitado todavía.
- **Gate P3:** beta privada de investigación permitida solo tras controles legales/privacidad; sin productos BET ni cobro público.

### P4 — Suscripción y mejora continua

- [ ] Verificar Stripe checkout, webhooks, FREE->PRO, cancelación, renovación, fallos y reembolso en sandbox; \`SOCCER_PUBLIC_BILLING_ENABLED=false\` hasta aprobación.
- [ ] Historial de decisiones congeladas, errores, settled bets reales, ROI/CLV por familia, n y versión; separar OOS de ROI.
- [ ] Entrevistas de retención: ¿qué vería el usuario 3 veces por semana?, ¿pagaría por filtros/alertas/profundidad?, ¿por qué cancelaría?
- [ ] Lanzamiento por jurisdicción y derechos confirmados, no como una apertura simultánea de todas las familias.
- **Gate P4:** firma de producto/seguridad/legal/finanzas y familia de mercado aprobada; **cero BET no constituye fallo**.

## 8. KPI de negocio propuestos, no performance inventada

- Descubrimiento: porcentaje de usuarios que abre Today y llega al Match Center.
- Valor recurrente: usuarios activos semanalmente, partidos guardados por usuario, revisitas pre-kickoff.
- Calidad del radar: porcentaje de candidatos con datos as-of suficientes, tiempo hasta detectar invalidaciones y número de WATCH por causa.
- Confianza: porcentaje de pronósticos guardados antes del inicio, porcentaje con resultado FT posterior, Brier y calibración por familia y fecha.
- Delivery: latencia de refresh, quota/cap, fallos de proveedor, falsas alertas y click-through de notificaciones consentidas.
- Monetización (solo después de autorización): conversión de FREE a PRO, churn, renovación, soporte y devoluciones.
- **Prohibido:** reportar ROI de research como ROI de apuestas; prometer cuota rentable a todos los usuarios.

## 9. Priorización / próxima decisión

**Orden recomendado de trabajo:**
1. **P0: cubrir y reconciliar el screening de todos los elegibles.** La última ejecución hizo 8/14, no 14/14; sin este paso no se puede prometer radar integral.
2. **P1: hacer confiables los modelos 1X2/FT goals/BTTS y abrir el track independiente de corners.** Un catálogo con todos los mercados no equivale a un análisis deportivo de cada uno.
3. **P3 en paralelo: acceso privado seguro a las ocho RAW refinadas y UI transparente de investigación.** No activar billing ni alertas BET.
4. **P2: precios reales y validación específica de cada mercado.** Primero demostrar precio exacto posterior al RAW, luego shrinking/EV y OOS, por separado.
5. **P4: auditar el flujo de suscripción y testear demanda antes del cobro.**

### Semáforo honesto al 2026-11-10

- **VERDE: infraestructura de discovery, motor RAW limitado, guardado privado prospectivo, preview frontend y controles básicos de acceso (implementados).**
- **ÁMBAR: capacidad de análisis de cada fixture elegible, profundidad de inputs, cobertura de varias familias y producto de investigación que el usuario puede usar (parcial/en QA).**
- **ROJO: cobertura completa de todos los mercados, validez BET de cualquier familia, precio directo ejecutable, ROI acreditado, alertas accionables, derechos/datos y cobro público (no verificados o bloqueados).**

**Conclusión:** estamos cerca de una *beta útil de inteligencia de fútbol* si priorizamos la conexión segura de datos y la cobertura por partido; estamos **sensiblemente más lejos de "todos los partidos, todos los mercados completos y picks BET"**, porque ese objetivo necesita nuevos modelos deportivos por mercado, más cobertura estadística y capacidad, precios ejecutables y suficientes observaciones verdaderamente prospectivas. No asignar porcentaje global de progreso ni fecha de lanzamiento sin SLA, coste/coverage y muestra OOS reales.

## 10. Fuentes de control / no-regresión

1. \`12_SOCCER_EDGE_ENGINE_MASTER_v1.0.md\` — SPORT FIRST, tres screens y thresholds de madurez.
2. \`11_SOCCER_DATA_API_PROTOCOL_v1.0.md\` — cobertura, endpoint map, source-as-of y odds.
3. \`17_SOCCER_FRONTEND_PRODUCT_ROADMAP_v1.0.md\` — FE-0–FE-11, FREE/PRO, Stripe, PWA y notificaciones.
4. \`docs/SOCCER_COMMERCIALIZATION_LAUNCH_GATES_v1.md\` — gates comerciales y legales.
5. \`soccer-edge-state:soccer_edge_state/analysis/soccer_commercialization_readiness_v1.json\` — auditoría cuantitativa dinámica; actualizar antes de decisiones futuras.
6. [GitHub Actions #42891660125](https://github.com/Baahl11/Soccer/actions/runs/42891660125) — 316 fixtures / 14 Tier A/B / 8 RAW / 35 provider calls / 8 PG rows ACK.
7. [API-Football V3](https://www.api-football.com/documentation-v3), [API-Football Terms](https://www.api-football.com/terms) — datos publicados por el proveedor y límites de licenciamiento.

**Version control:** \`PRODUCT_OFFER_AND_FULL_SLATE_MARKET_READINESS_v1.0\`; documento de estrategia/progreso, **NO** nueva versión del modelo Soccer. Los planes futuros requieren aprobación; las observaciones pasadas son inmutables.
