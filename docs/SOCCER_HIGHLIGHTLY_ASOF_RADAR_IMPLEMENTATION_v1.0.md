# Soccer Edge — Highlightly as-of + Radar privado v1.0

**Fecha de observación:** 2026-11-10, México central / 2026-11-11 UTC.

**Estado:** Adaptador de investigación, no sistema comercial. Futuros usos como input cuantitativo quedan bloqueados hasta validación OOS y de licencias.

## Resultado real de la primera prueba

Run GitHub Actions: https://github.com/Baahl11/Soccer/actions/runs/42963087908 (SUCCESS).

Se usaron **6 consultas** de Highlightly BASIC con 12 fixtures independientes del Soccer Edge API / API-Football v3; 8 identidades quedaron verificadas contra nombres de ambos equipos, hora de kickoff UTC ±15 minutos, país, competición, temporada e IDs existentes. Una discrepancia de nombre de liga quedó en LEAGUE_NOT_VERIFIED, tres referencias no aparecieron en las páginas consultadas. Es una muestra de conveniencia, no la cobertura global del proveedor.

Para el encuentro futuro Chernihiv–Inhulets (fixture canónico 1581906, 2026-11-11 09:30 UTC), se recogieron estadísticas de temporada por ambos equipos **antes del inicio** (05:02 UTC). Dos respuestas HTTP 200, una entrada de la liga y temporada por equipo, familia FORM presente. Fue una captura de investigación; no puede reconstruir input de predicciones ya cerradas, ni acredita source_updated_at del proveedor, ni demuestra xG o todas las familias de mercado. Cuota restante al final de esta prueba: 88/100 (dato as-of).

## Conexión segura implementada

- Un nuevo script principal llama Highlightly únicamente desde GitHub Actions y lee la clave protegida SOCCER_HIGHLIGHTLY_API_KEY sin imprimirla. Tope 10 solicitudes y stop con 70 restantes.
- Mapeo exacto sin coincidencias fuzzy, registros y horas de captura, flags por familia con ambas entidades presentes; fútbol únicamente; nunca consulta cuotas.
- GitHub OIDC firmado al endpoint interno de Render, exclusivo de workflow en rama main.
- Tabla independiente de investigación en Render PostgreSQL: IDs de fixtures canónicos, estado, familias, timestamps y hashes; no almacena respuestas proveedor ni modifica tablas principales.
- /app/api/v2/today incorpora información complementaria al Radar **solo para owner/admin**. FREE/PRO público no recibe atributos externos hasta que se acrediten derechos de visualización. Si falta dato, NOT VERIFIED, nunca cero ni BET.
- No cambia probabilidades RAW, cuotas, clasificaciones del motor, OOS, CLV, Stripe, user entitlements o lógica de negocio.

## Gaps restantes antes del servicio PRO

1. Completar screening del 100% de fixtures elegibles o registrar cada exclusión/pendiente; escalar caché, queue y control de presupuesto.
2. Verificar datos por familia, liga y momento pregame (especialmente xG, córners, tiros, tarjetas y minutos de jugador). Validar cada muestra histórica para que no contenga datos posteriores al kickoff.
3. Calibrar modelos SPORT independientes y medir OOS/FT para cada mercado. Córners/cards/player props/SGP no son familias BET validadas.
4. Capturar precio real ejecutable y fresco por bookmaker y selección después del RAW; devig, shrink, EV, settlement, CLV verdadero. Highlightly BASIC no sustituye ese acceso.
5. Derechos de redistribución, UX, RLS, autenticación, privacidad/jurisdicción y pruebas Stripe sandbox. Cobro público NO-GO.
6. Sistema de alertas consentidas con invalidación, deduplicación y fuente actualizada, solo tras disponibilidad y mercado verificados.

**Fuentes:** https://highlightly.net/football-api/documentation/ y https://highlightly.net/football-api/documentation/getting-started/pagination/ .

**Versión:** HIGHLIGHTLY_ASOF_RADAR_INTEGRATION_v1.0 (investigación, sin cambiar versión del modelo Soccer).

## End-to-end OIDC resultado real, 2026-11-10 23:14 México central

La ejecución de main [GitHub Actions #42964423528](https://github.com/Baahl11/Soccer/actions/runs/42964423528) terminó SUCCESS y recibió respuesta firmada de la ruta de Render ya desplegada (commit runtime 14bd6ea). Informe de recibo [artifact #13830183850](https://github.com/Baahl11/Soccer/actions/runs/42964423528/artifacts/13830183850).

- Fuente HIGHLIGHTLY_BASIC_RESEARCH_ONLY, 6 GET, remaining **82/100** inmediatamente al último request.
- 12 identidades evaluadas, 8 matches estrictos, 1 registro con familia FORM capturada antes del partido.
- **ACK de PostgreSQL recibido:** records_received=12, rows_inserted=12, status VERIFIED_PRIVATE_RESEARCH_IMPORT.
- El protocolo OIDC admitió exclusivamente el workflow main identificado. No se transmitieron resultados estadísticos numéricos, cuotas, key ni datos personales; únicamente estados, familias y timestamps con flags comerciales false.
- Runtime /app-v2 dispone de lectura **owner/admin only** de esa tabla privada a través de /app/api/v2/today. No se verificó una sesión real del propietario mediante navegador; el release de Render y los tests de contrato sí pasaron. No reclamar QA visual de cuenta como completado.
- La herramienta de consultas externas a Render PostgreSQL no puede acceder a esta base, porque el allowlist de IP está vacío. **No abrir acceso externo para comprobarlo**; se mantiene como evidencia el ACK del servidor autenticado y su resultado de inserción, no una consulta independiente directa.

**Cierre operacional:** el workflow vuelve a modo manual-only, sin schedule ni trigger push, tras esta única corrida productiva de investigación. Cualquier futura repetición requiere ejecución explícita y budget verificado. No es un servicio diario de screening total.

## Estado comercial y próximo cuello de botella

La importación de 12 registros de investigación NO significa 12 deep dives completos ni 12 predicciones validadas: solo estado de identidad/calidad de una segunda fuente. Después de esto las siguientes prioridades son **1) screening deportivo de todos los elegibles con colas y caché, 2) datos históricos pre-kickoff comparables por familia y OOS, 3) modelos específicos de córners/cards/player y disponibilidad XI, 4) precios directos ejecutables/EV/CLV, 5) derechos de datos, QA de autenticación/RLS, Stripe sandbox y cumplimiento legal**. Seguimos en RESEARCH/NO-GO comercial.
