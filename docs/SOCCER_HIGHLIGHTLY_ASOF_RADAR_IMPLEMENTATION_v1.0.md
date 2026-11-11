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