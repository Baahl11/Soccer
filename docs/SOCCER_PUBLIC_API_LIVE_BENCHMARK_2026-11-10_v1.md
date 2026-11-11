# SOCCER EDGE — Benchmarks de APIs públicas realmente ejecutados v1.0

**Fecha:** 10 de noviembre de 2026, America/Mexico_City  
**Ejecución UTC:** 2026-11-11 04:46  
**Modo:** RESEARCH ONLY. 0 claves privadas, 0 compras, 0 cuotas deportivas consultadas, 0 escrituras de bases canónicas, 0 BET.  
**CI definitivo:** [GitHub Actions #42961413372](https://github.com/Baahl11/Soccer/actions/runs/42961413372), SUCCESS.  
**CI inicial:** [#42961248208](https://github.com/Baahl11/Soccer/actions/runs/42961248208), SUCCESS.
**Script:** [soccer_public_free_provider_probe_v1.py](../.github/scripts/soccer_public_free_provider_probe_v1.py).
**Workflow:** [soccer-public-free-provider-probe-v1.yml](../.github/workflows/soccer-public-free-provider-probe-v1.yml), MANUAL ONLY, sin push ni cron luego de dos ejecuciones.

## Protocolo y límites

- Se identificaron **12 partidos reales** de noviembre 9, 10 y 11 mediante Soccer Edge API / API-Football v3 a las 2026-11-11T04:41Z. Incluyen ligas de España, Portugal, Italia, México, Brasil, UEFA Femenina, Rumanía, Inglaterra U21, Ucrania y selecciones juveniles. Selección por conveniencia: **NO representativa** del universo de 316 partidos ni de todas las ligas de cada proveedor.
- 13 GET HTTPS contra endpoints de demo/gratis declarados oficialmente, con timeout, cap duro y sin retries. Nunca se usaron credenciales pagadas ni datos de otras cuentas.
- Match cross-provider solamente por nombres normalizados exactos y UTC kickoff con ±15 minutos. Puede haber falsos negativos por alias/zonas horarias/cobertura restringida. **Cero matches verificados en planes demo NO es cobertura comercial cero.**
- El JSON del proveedor no se publica; solo se exponen metadatos de consultas, cantidades y familias de campos presentes. No hay una predicción prepartido, OOS ni valoración de apuestas.

## Resultado real del segundo probe

| Proveedor y alcance | HTTP | Resultado | Coincidencia estricta en 12 | Interpretación |
|---|---|---|---|---|
| **OpenFootAPI demo** | 200 para tres fechas, catálogo y control; 403 para xG | Nov9 **5**, Nov10 **5**, Nov11 **1**; catálogo **120** competiciones | **0/12** en plan demo | Acceso gratuito probado; xG NO disponible en demo, aun siendo anunciado en plan Developer |
| **TheSportsDB clave pública 123** | 200 en 4 calls | Nov9 **3**, Nov10 **3**, Nov11 **3**; control Arsenal **1** | **0/12** en free limitado | Tier recorta listados y no demuestra capacidad para análisis profundo del universo |
| **OpenLigaDB sin clave** | 200 | Jornada Bundesliga alemana: **9** partidos | No comparable | Útil de apoyo de calendario/resultados y licencias ODbL; no muestra xG global |
| **FootyStats ejemplo público** | 200 en league-teams y league-season | Datos históricos Premier League 2024/25; ambas respuestas contenían campos no vacíos de **xG, córners, tiros, tarjetas y posesión** | **NO MEDIDO** para noviembre 2026 | Mejor prueba pública de profundidad estadística real; acceso a las ligas/temporadas actualizadas requiere clave propia |
| **Highlightly** | NO CONSULTADO: clave privada obligatoria | Solo afirmaciones de documentación | NOT VERIFIED | No inventar su coverage, pedir free key para benchmark |
| **Sportmonks** | NO CONSULTADO: clave privada obligatoria | Solo afirmaciones de documentación | NOT VERIFIED | No es sustituto de alto volumen con plan básico de cinco ligas |

**Total real: 13 requests.** El primer probe hizo 11 y el segundo sumó dos para FootyStats, sin persistencia de payload ni llamadas a odds. El segundo ejecuto de nuevo los 11 GET públicos, por lo que el conteo por corrida es 13 y las corridas son distintas. No llamar "26 requests de la segunda corrida".

**No es una evaluación válida de qué API da más datos en Liga MX**, ni de xG disponible antes del kickoff. La API FootyStats histórica confirma campos poblados, pero no su oportunidad/frescura para proyectar pregame. OpenFoot responde 403 en xG con demo, sin probar Developer.

## Qué credenciales harían falta para una comparación equivalente

**Nada nuevo para lo que ya medimos.** Para probar una muestra estratificada de al menos 30 partidos idénticos y comparar xG/córners/tarjetas/tiros/jugadores/lineups por liga:

1. **Highlightly:** crear una cuenta gratuita desde https://highlightly.net/football-api/ y obtener su API key. Guardarla solamente en GitHub Actions Secrets, Render Secret Environment o gestor de credenciales; nombre propuesto SOCCER_HIGHLIGHTLY_API_KEY. No publicarla en chats, commits o issues.
2. **OpenFootAPI:** Starter gratis sirve para fixtures, NO para xG; acceso Developer **US$14/mes** o un dataset de muestra xG autorizado por el proveedor para top-five. Requiere consentimiento antes de compra. Nombre propuesto SOCCER_OPENFOOT_API_KEY.
3. **FootyStats:** clave propia para acceso a Liga MX y temporada actual, porque clave example es demo de Premier League 2024/25. Solicitar confirmación de que el plan incluye ligas y parámetros buscados. Nombre propuesto SOCCER_FOOTYSTATS_API_KEY.
4. Sportmonks es opcional: comparar solo si interesa comprar profundidad en cinco ligas.

No solicitamos ni copiamos secretos aquí. Registrar cuentas, aceptar TOS y pagar requiere la acción explícita del titular.

## Gate de adopción

- Comparar los mismos **≥30 fixtures**: Liga MX/LATAM, cinco grandes europeas, ligas de menor cobertura.
- Verificar identidad con fixture id, competición, home/away, kickoff UTC y observación original, nunca por nombre fuzzy únicamente.
- Medir tasas por familia: xG shot-based real, corners for/against, cards, SOT/shots, player minutes, XI, tamaño de muestra pregame y su as-of disponible. Campo faltante = NOT VERIFIED, NO cero.
- Atribuir datos, permisos de redistribución para uso de producto y restricciones de origen. OpenFootAPI permite investigación interna y almacenamiento pero no licencia la redistribución por sí mismo.
- Medir coste **por partido-deporte-modelable**, no por precio del plan ni número anunciado de ligas. Controlar quota, caché y latencia.
- No mezclar probabilidades derivadas por proveedor ni fair odds con RAW SPORT; no recalcular historia congelada ni promover BET hasta OOS y market gate.

**Decisión:** mantener API-Football canónica por ahora. Para mejorar riqueza de xG y mercados, la siguiente prueba autorizable y económica es clave gratis de Highlightly para cobertura y, con consentimiento, prueba acotada de OpenFoot Developer y/o acceso a FootyStats de ligas concretas. Ninguna compra ni integración productiva quedó activada.

## Fuentes oficiales
- [OpenFoot demo y documentación](https://openfootapi.com/docs); [precios](https://openfootapi.com/pricing); [licencia de origen](https://openfootapi.com/terms).
- [TheSportsDB free docs](https://api.thesportsdb.com/documentation).
- [OpenLigaDB API](https://api.openligadb.de/index.html); [ODbL](https://openligadb.de/).
- [FootyStats clave example](https://footystats.org/api/documentations/try-all-endpoints); [FAQ](https://footystats.org/api/documentations/faq).
- [Highlightly auth](https://highlightly.net/football-api/documentation/).
- [Comparación de planes previa](SOCCER_ALTERNATIVE_DATA_PROVIDERS_BENCHMARK_v1.0.md).

**Versión:** PUBLIC_FREE_PROVIDER_BENCHMARK_v1.0. Documentación de investigación, no cambio de versión de motor ni picks.
