# Soccer Edge — checklist de cierre del Expediente deportivo

**Alcance:** cada matchup elegible en el frontend OWNER/admin, reutilizando el registro canónico, PostgreSQL y los modelos existentes. Un partido con datos incompletos debe mostrar su estado y causa; «100 %» significa trazabilidad, no 100 % de partidos con pronóstico ni autorización de apuestas. SPORT FIRST, MARKET SECOND.

## 1. Base y despliegue
- [x] Reconciliar alcance del scanner y FE11 sin promover mercados (#189, #190).
- [x] Verificar un lote de 3 perfiles FE11 exportados e insertados en PostgreSQL (Actions 43432746878; segundo lote 43470612105).
- [x] Mostrar el Expediente deportivo en el frontend OWNER/admin (#192).
- [x] Filtrar por fuente y tiempo **antes** de elegir la observación prepartido más reciente (#193). Render verificado en Actions 43473070534.
- [x] Integrar lectura directa del RAW original en PostgreSQL, con identidad y digest, y tasas de gol por equipo sin presentarlas como xG ni como probabilidades por línea (#194, diez checks).
- [x] Render verificó el commit del expediente y HTML servido (Actions 43475700873). La comprobación independiente de acceso privado sin autorización pasó en 43475566312. El HTML no sustituye la prueba de sesión autenticada.

## 2. Prueba real PostgreSQL → match API → navegador
- [x] Lote FE11 del 13 de noviembre de 2026 (Actions 43470612105): tres perfiles exportados y tres inserciones confirmadas. Captura común `2026-11-13T17:14:03.578710+00:00`.
- [x] Cerrar la prueba de visualización OWNER del lote: 3 exportados → 3 insertados → 3 leídos y visibles en su ficha. Las capturas coinciden en equipos, fuente temporal y muestras; no muestran el kickoff ni la URL, identificados con el artefacto y las rutas indicadas para la prueba. No se archivaron capturas privadas ni credenciales.
  - [x] `1647362` Argentina–Uruguay: captura OWNER confirma el perfil FE11, con los mismos equipos y timestamp `2026-11-13T17:14:03.578710+00:00` del artefacto del scanner 43470612105. Muestras por localía n=3 por lado, etiquetadas MUESTRA BAJA. Solo histórico; no RAW ni OOS.
  - [x] `1640518` Colombia–Chile: perfil FE11 visible, mismo timestamp, n=3 local y n=4 visitante.
  - [x] `1647364` Perú–Bolivia: perfil FE11 visible, mismo timestamp, n=0 local (sin dato) y n=1 visitante (los 0.00 goles observados no equivalen a un campo ausente).
  - [x] Aclarar el alcance del contador: el texto de #198 dice «Observaciones prepartido guardadas» y aclara que FE11 se muestra aparte.
- [ ] Verificar un partido con RAW PostgreSQL y otro sin RAW. Probar enlace directo o recarga sin depender del listado del día. Confirmar que un usuario no OWNER no recibe los campos privados añadidos. Una prueba con datos de ejemplo no cierra este paso.

## 3. Completar las familias en esa misma ficha, sin recolectores paralelos
- [x] Goles por equipo: lectura privada por lado y media línea, separada de λ (#198, Render 43482010018).
- [ ] Goles por equipo: verificar una ficha OWNER real con el grid guardado y otra sin grid.
- [x] Córners FT: mostrar tasas guardadas de local, visitante y total con procedencia, separadas del FE11 (#199, Render 43483666414).
- [ ] Córners FT: verificar una ficha OWNER con tasas guardadas; conectar salidas por línea solo cuando exista registro verificable y se conserve la semántica de liquidación.
- [ ] 1H/2H: FE11 ya es visible como histórico. No reutilizar el modelo condicionado al marcador del descanso como proyección previa al kickoff; falta confirmar proyecciones prepartido guardadas por periodo.
- [x] Amarillas FT: mostrar tasa total prepartido guardada separada de rojas/puntos (#200, Render 43483955893).
- [ ] Verificar una ficha OWNER con la tasa de amarillas guardada. Líneas, tarjetas por equipo y rojas se aceptan por separado solo con salidas verificadas.
- [ ] Props: verificar fixture, player_id, XI/minutos y procedencia prepartido; conectar tiros/SOT primero, luego demás familias cuando tengan evidencia. No reactivar backfills bloqueados por memoria sin prueba de capacidad.
- [ ] Para cada familia mostrar por separado evidencia, RAW, OOS y precio; distinguir NO VERIFICADO, muestra insuficiente, aplazado y diferido por cuota. Mantener muestra, fuente y hora al lado del dato.

## 4. Continuidad y aceptación
- [ ] Reconciliar por competición y familia cada fixture elegible (calculado, evidencia solamente, pendiente o bloqueado); medir diferencias entre detectado, importado, leído y mostrado, no solo éxito de Actions.
- [ ] Verificar refrescos y resultado oficial sin alterar la proyección congelada; evaluar OOS por familia y versión, sin extrapolar.
- [ ] Mantener BET comercial bloqueado mientras no pasen las puertas independientes de OOS, disponibilidad, precio ejecutable, liquidación y derechos. No se necesita abrir BET para dar por terminada la integración del expediente.

Marcar un punto solo con un resultado verificable y registrar fecha UTC, SHA y ejecución o prueba correspondiente.
