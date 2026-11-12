# Soccer Edge — implementación total de inteligencia deportiva (2026-11-11)

**Principio:** 100% de partidos elegibles examinados; proyecciones solo cuando estén respaldadas; 0 recomendaciones de valor es válido. Ningún PASS oculta el análisis. BET/LEAN/WATCH/PASS sigue siendo la clasificación CANÓNICA interna (no se reescribe el ledger).

## Checklist de trabajo

- [x] **PR 1 — Ficha universal y contrato independiente.** Cada fixture del registro expone `sport_analysis` y `opportunity`, con estados separados y razones; se preservan PASS, WATCH, DATA INSUFFICIENT, MARKET_NOT_EVALUATED. No hay datos inventados.
- [x] **PR 2 — Cola persistente de cobertura (estructura y pruebas).** `soccer_sport_screen_queue` idempotente, alimentada por `DAILY_DISCOVERY`, con estado y fecha del último screening; pendientes no se marcan como analizados. Turnos siguientes reintentan con cuota reservada.
- [x] **PR 3 — Contrato de proyección / factores verificables en su alcance permitido.** Tres tracks SIDE/GOALS_BTTS/CORNERS, con `RAW` y provenance, `NOT VERIFIED` explícito para familias faltantes; histórico inmutable.
- [x] **PR 4 — Recomendaciones independientes (presentación y clasificación).** Mapper visual de los cuatro estados internos; `MARKET_NOT_EVALUATED` si no hubo verificación, `NO_VALUE` únicamente con PASS sustentado. Precio, fuente, timestamp, availability y EV permanecen sujetos a gates.
- [x] **PR 5 — Cuaderno del analista (Supabase RLS aplicada y backend/frontend implementados).** Notas por usuario/fixture, sin afectar modelos, protegidas por autenticación + RLS; persistencia y prueba de aislamiento.
- [x] **QA — Unitarias, seguridad, integración y PostgreSQL desechable (15/15 CI aprobadas en rama de trabajo).** Pruebas Python, JS compilable, fixture solo con registro, PASS con datos, PRICE missing, disponibilidad sin verificar, cola reentrante, OOS/licencia, notas. La validación del deploy y del tick está separada.
- [x] **Permisos de visualización separados en código.** Separar permiso de ver resultados técnicos de permiso de redistribuir probabilidades/datos del proveedor. El primer permiso no implica el segundo.

## Puertas de liberación operativa (no cerrar por un merge)

- [ ] Render sirve exactamente el commit de integración y el frontend usa `sport_analysis` + `opportunity`.
- [ ] El ciclo canónico guarda `soccer_sport_screen_queue` con todos los fixtures elegibles, más contadores P0 reales y sin fallos de persistencia.
- [ ] Auditar la visualización móvil real y la ruta autenticada de notas (no sólo RLS y mocks).
- [ ] Cobertura deportiva progresiva del 100% de fixtures elegibles antes de sus kickoffs, medida en varias jornadas; no confundir `QUEUED` con `MODELED_LIMITED`.
- [ ] Validación OOS por familia y permisos de redistribución para proyecciones publicables.
- [ ] Revisión jurídica de copy y uso comercial de cuotas/recomendaciones antes del lanzamiento público.

## Criterios de resultado

1. Un partido con PASS **mantiene su ficha deportiva**, sus fuentes y sus limitaciones.
2. Un partido sin cuotas verificadas **no** se etiqueta “sin valor”; dice **MERCADO NO EVALUADO**.
3. Un partido con un solo dato conocido no se declara analizado: estado **DATOS INSUFICIENTES** o **PARCIAL**.
4. Los 15 partidos de una jornada de ejemplo tendrán estados, no necesariamente 15 pronósticos.
5. Vista anónima: solo datos autorizados; vista con permiso: profundidad según entitlement; no abrir investigación OOS/licencia pendiente a FREE.
6. Cada ejecución registra intentos, resultados, motivos y cuota; todos los estados trazables por fixture, sin odds-first.
7. Ninguna observación humana altera un peso, predicción, precio o clasificación histórica automáticamente.

## Riesgos que bloquean la publicación completa de predicciones y recomendaciones
- Live P0 solo puede procesar lo que permitan datos/caché y cuota; no prometemos 100% de pronósticos.
- OOS y derechos de redistribución pendientes bloquean publicación de modelos de investigación.
- El deploy a Render y los checks de GitHub se deben verificar por separado; no basta un merge.
