# FE-10 — Contrato de alertas móviles Soccer Edge

**Versión:** diseño v0.1 | **Fecha:** 2026-11-06 | **Estado:** PLAN, sin entrega push habilitada.

## Objetivo

Avisar a un suscriptor que lo solicite cuando el motor canónico detecte una oportunidad cuyo precio real sea **-110 o mejor**, incluyendo +100 y precios positivos superiores. La comparación es decimal `price >= 21/11` (>= 1.9090909). La cuota sola **no es una razón de apostar**.

El proceso comienza con SPORT FIRST y una predicción RAW independiente; solo después compara oferta, vig y la probabilidad conservadora. Nunca fabricar probabilidades, precio, XI, lesiones ni valor.

## Disparadores seguros

**BET_ALERT:** exige `classification=BET` del motor canónico vigente, versión aprobada para ese mercado, línea/selección exactas, fuente/casa, cuota accionable verificada, quote timestamp reciente, precio aún válido, kickoff futuro, estado no suspendido, disponibilidad esencial confirmada y controles de riesgo. Sin cualquiera de estos campos no disparar BET.

**RESEARCH_RADAR:** puede usarse en fase beta para `LEAN/WATCH`, separado y con consentimiento específico; nunca etiquetar como BET ni sugerir que el precio garantice ventaja. Mostrar qué falta por verificar. Evitar avisos de mera cuota par sin señal deportiva fundamentada.

Para Bet365 ID8 de API-Football: es **precio de referencia**, no una oferta demostrada en la cuenta del usuario. Una notificación debe solicitar reconfirmación manual en la casa y no afirmar que la cuota fue aceptada.

## Carga de evento

```text
event_id, candidate_id, fixture_id, kickoff_utc, league_id,
market_family, market_key, selection, exact_line,
bookmaker_id, bookmaker_name, source, price_decimal,
provider_updated_at, captured_at, expires_at,
raw_probability, market_shrunk_probability,
fair_market_probability, breakeven_probability,
probability_edge_pp, estimated_ev,
availability_confidence, lineup_status, goalkeeper_status,
classification, tier, model_version, decision_version,
sporting_reasons[], market_reasons[], blockers[],
event_type, occurred_at
```

Campos ausentes deben permanecer NULL/NOT_VERIFIED, jamás cero por defecto. EV calculado solo para mercados cuyo settlement se modele correctamente, incluidos push y líneas asiáticas.

## Diseño de experiencia

- **App web:** panel de alertas con consentimiento; tarjeta de partido, mercado, selección, línea, cuota exacta, actualización, estado y caducidad.
- **PWA/Web push:** permisos opt-in; no activar si el navegador/usuario no da permiso. Servidor VAPID, persistencia de suscripciones, rotación/revocación y entregas seguras.
- **Aplicación móvil futura:** FCM/APNs según plataforma; mismo motor, distinta infraestructura de entrega.
- **Preferencias:** ligas, mercados, cuota mínima (por defecto -110 o mejor), solo BET o research, zona horaria, horas de silencio, desactivación inmediata.
- **Dedupe/antispam:** identidad de fixture+mercado+selección+línea+casa+versión de cuota y clasificación; enfriar avisos repetidos; no reenviar sin cambio material.
- **Anulación:** enviar `PRICE_MOVED/INVALIDATED/CANCELLED` para cerrar alertas obsoletas; nunca borrar historial original.
- **Seguridad:** no exponer tokens API, apuestas del usuario ni datos sensibles en notificación de pantalla bloqueada; jurisdicción/edad/copy legal bajo revisión.
- **Acción:** abrir Match Center con validación de precio; ninguna apuesta se coloca automáticamente y no se promete rentabilidad.

## Criterios de aceptación antes de habilitar push

1. Pruebas de opt-in/opt-out, revocación, RLS, acceso FREE/PRO, desconexión, expiración TTL y horarios.
2. Dedupe probado ante eventos T-90/T-40/T-20, cambios de cuota y reintentos.
3. Alineaciones/portero e información material para BET correctamente verificadas por el motor.
4. Ausencia de avisos al comenzar el partido, sin mercado válido o con fecha stale.
5. Pruebas Android/iOS PWA; rate limit razonable y medición de entregas (sin contaminar el modelo de apuestas).
6. Avisos de research claramente separados de apuestas aprobadas.
7. Evidencia prospectiva y precio ejecutable antes de comercializarlo como beneficio de picks.

## Orden de construcción

**FE-10A** preferencias y event/outbox privados → **FE-10B** notificaciones dentro de App V2 con pruebas → **FE-10C** PWA push con consentimiento → **FE-10D** app nativa si la demanda y pruebas justifican.

**Estado actual:** no existen entregas push, notificaciones automáticas de apuestas ni billetera/ejecución de apuestas. Esta especificación no los habilita.
