# Soccer Edge — implementación total de inteligencia deportiva (2026-11-11)

**Principio:** 100% de partidos elegibles examinados; proyecciones solo cuando estén respaldadas; 0 recomendaciones de valor es válido. Ningún PASS oculta el análisis. BET/LEAN/WATCH/PASS sigue siendo la clasificación CANÓNICA interna (no se reescribe el ledger).

## Checklist de trabajo

- [ ] **PR 1 — Ficha universal y contrato independiente.** Cada fixture del registro expone `sport_analysis` y `opportunity`, con estados separados y razones; se preservan PASS, WATCH, DATA INSUFFICIENT, MARKET_NOT_EVALUATED. No hay datos inventados.
- [ ] **PR 2 — Cola persistente de cobertura.** `soccer_sport_screen_queue` idempotente, alimentada por `DAILY_DISCOVERY`, con estado y fecha del último screening; pendientes no se marcan como analizados. Turnos siguientes reintentan con cuota reservada.
- [ ] **PR 3 — Proyección / factores verificables.** Tres tracks SIDE/GOALS_BTTS/CORNERS, con `RAW` y provenance, `NOT VERIFIED` explícito para familias faltantes; histórico inmutable.
- [ ] **PR 4 — Recomendaciones independientes.** Mapper visual de los cuatro estados internos; `MARKET_NOT_EVALUATED` si no hubo verificación, `NO_VALUE` únicamente con PASS sustentado. Precio, fuente, timestamp, availability y EV permanecen sujetos a gates.
- [ ] **PR 5 — Cuaderno del analista.** Notas por usuario/fixture, sin afectar modelos, protegidas por autenticación + RLS; persistencia y prueba de aislamiento.
- [ ] **QA — End to end.** Pruebas Python, JS compilable, roles FREE/PRO/OWNER, fixture solo con registro, PASS con datos, PRICE missing, disponibilidad sin verificar, cola reentrante, OOS/licencia, notas; comprobar commit servido por Render y tick real antes de considerar LIVE.
- [ ] **Publicación.** Separar permiso de ver resultados técnicos de permiso de redistribuir probabilidades/datos del proveedor. El primer permiso no implica el segundo.

## Criterios de resultado

1. Un partido con PASS **mantiene su ficha deportiva**, sus fuentes y sus limitaciones.
2. Un partido sin cuotas verificadas **no** se etiqueta “sin valor”; dice **MERCADO NO EVALUADO**.
3. Un partido con un solo dato conocido no se declara analizado: estado **DATOS INSUFICIENTES** o **PARCIAL**.
4. Los 15 partidos de una jornada de ejemplo tendrán estados, no necesariamente 15 pronósticos.
5. Vista anónima: solo datos autorizados; vista con permiso: profundidad según entitlement; no abrir investigación OOS/licencia pendiente a FREE.
6. Cada ejecución registra intentos, resultados, motivos y cuota; todos los estados trazables por fixture, sin odds-first.
7. Ninguna observación humana altera un peso, predicción, precio o clasificación histórica automáticamente.

## Riesgos que bloquean un DONE real
- Live P0 solo puede procesar lo que permitan datos/caché y cuota; no prometemos 100% de pronósticos.
- OOS y derechos de redistribución pendientes bloquean publicación de modelos de investigación.
- El deploy a Render y los checks de GitHub se deben verificar por separado; no basta un merge.
