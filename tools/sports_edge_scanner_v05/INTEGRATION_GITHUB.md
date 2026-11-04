# SPORTS EDGE SCANNER v0.5 — integración aislada en `Baahl11/Soccer`

**Estado:** Propuesta en rama / PR para revisión. Es un módulo separado y **no reemplaza** el scheduler, el modelo v4 ni el estado canónico del repositorio.

## Qué se instala

- Código original v0.5, bajo `tools/sports_edge_scanner_v05/sports_edge/`.
- Sus 56 pruebas unitarias offline y archivos de ejemplo.
- Un workflow dedicado `.github/workflows/sports-edge-scanner-v05-paper.yml`.
- La ejecución en pull request **solo ejecuta pruebas sin usar API**.
- La ejecución `workflow_dispatch` es **manual** y opera en modo `WATCH`/research; no genera apuestas ni actualiza modelos.

## Por qué no se activa cron todavía

El repositorio `main` tiene un scheduler real y un commit del 13-oct-2026 que pausó 28 trabajos programados por restricciones del proveedor. Encender otro calendario sin revisar el presupuesto común y los datos redundantes puede duplicar consultas. **Este workflow no cambia el scheduler actual** ni habilita un segundo cron.

## Activación segura

1. Revisar y combinar la PR; GitHub solamente permite despachar manualmente workflows existentes en la rama predeterminada.
2. En `Settings → Secrets and variables → Actions`, crear/confirmar un secreto llamado `API_FOOTBALL_KEY` usando la clave de tu cuenta Pro. **Nunca compartirla en el chat ni subirla a Git**. Esta integración no puede leer ni crear los secretos del repositorio.
3. En `Actions → SPORTS EDGE SCANNER v0.5 - Soccer paper research → Run workflow`, ingresar `league_ids` **explícitos y aprobados por Soccer**, fecha opcional y `reference_odds_limit` de 0 a 25. Empezar con 0.
4. Ver el resultado y descargar el artefacto `soccer-paper-research`, disponible 7 días. Comprobar la auditoría `complete_slate_verified` antes de cualquier análisis posterior.

## Limitaciones actuales

- El proveedor API-Football ofrece cuotas de otros bookmakers; **no garantiza** cuotas accionables de Caliente o Playdoit.
- Las señales de goles y dominancia son **solo heurísticas de investigación**; no producen probabilidades calibradas, edge, EV ni BET.
- Las alineaciones/lesiones requieren verificación de fuente oficial. Vacío en endpoint no significa que nadie esté lesionado.
- El límite local de consultas se guarda dentro del runner y **no persiste automáticamente entre diferentes ejecuciones**; la cuota real del proveedor manda. No ejecutar en paralelo con el scheduler sin presupuesto compartido.
- No se automatizan casas de apuestas, apuestas, depósitos ni pagos.

## Validación local

```bash
cd tools/sports_edge_scanner_v05
python -m unittest discover -s tests -q
```

El README de la carpeta contiene los comandos `pro-soccer-daily`, `pro-soccer-refresh` y la política de WATCH/PASS.