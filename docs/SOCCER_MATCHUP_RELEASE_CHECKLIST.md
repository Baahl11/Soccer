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
- [x] Caso positivo OWNER: Colombia–Chile, fixture 1640518. Origen anterior al inicio en commit `30d0165d766a7cf03c178ee75de1d0f1d4c1e5c8`, captura `2026-11-12T22:52:23.128393+00:00`, mismo par ID/hora en el recibo de importación y valores coincidentes en ficha OWNER. No se archivan capturas ni credenciales.
- [x] Caso negativo visual OWNER: Perú–Bolivia, fixture 1647364, sin tarjeta RAW original en la posición esperada. Ausente en el lote congelado y en su recibo de 188 registros; no equivale a una consulta exhaustiva de toda la base.
- [ ] Verificación de acceso: probar respuesta de API sin sesión en Render y separación FREE/PRO/OWNER/ADMIN en contrato (#208). La prueba de contrato no sustituye una sesión comercial real en producción.

## 3. Completar las familias en esa misma ficha, sin recolectores paralelos
- [x] Goles por equipo: lectura privada por lado y media línea, separada de λ (#198, Render 43482010018).
- [ ] Goles por equipo: verificar una ficha OWNER real con el grid guardado y otra sin grid.
- [x] Córners FT: mostrar tasas guardadas de local, visitante y total con procedencia, separadas del FE11 (#199, Render 43483666414).
- [ ] Córners FT: verificar una ficha OWNER con tasas guardadas; conectar salidas por línea solo cuando exista registro verificable y se conserve la semántica de liquidación.
- [ ] 1H/2H: FE11 ya es visible como histórico. No reutilizar el modelo condicionado al marcador del descanso como proyección previa al kickoff; falta confirmar proyecciones prepartido guardadas por periodo.
- [x] Amarillas FT: mostrar tasa total prepartido guardada separada de rojas/puntos (#200, Render 43483955893).
- [ ] Verificar una ficha OWNER con la tasa de amarillas guardada. Líneas, tarjetas por equipo y rojas se aceptan por separado solo con salidas verificadas.
- [x] Tiros y tiros a puerta: integrar la lectura de salidas guardadas por jugador y línea, con fixture, XI/minutos y procedencia prepartido (#201, 10 comprobaciones).
- [ ] Confirmar despliegue de #201 y una ficha OWNER real con props guardadas; conectar las demás familias solo cuando tengan evidencia. No reactivar backfills bloqueados por memoria sin prueba de capacidad.
- [ ] Para cada familia mostrar por separado evidencia, RAW, OOS y precio; distinguir NO VERIFICADO, muestra insuficiente, aplazado y diferido por cuota. Mantener muestra, fuente y hora al lado del dato.

## 4. Continuidad y aceptación
- [x] Cruce exacto del calendario público con el diario congelado para 1X2, FT >2.5 y BTTS (#207, Actions 43670726320, Render `7336888`): 19 elegibles con identidad completa en 10 competiciones, sin coincidencias en ese corte.
- [ ] Extender la conciliación a importado, leído y mostrado por familia y distinguir evidencia, pendiente y bloqueado; el cruce público no certifica esas etapas.
- [ ] Verificar refrescos y resultado oficial sin alterar la proyección congelada; evaluar OOS por familia y versión, sin extrapolar.
- [ ] Mantener BET comercial bloqueado mientras no pasen las puertas independientes de OOS, disponibilidad, precio ejecutable, liquidación y derechos. No se necesita abrir BET para dar por terminada la integración del expediente.

Marcar un punto solo con un resultado verificable y registrar fecha UTC, SHA y ejecución o prueba correspondiente.

### Auditoría en producción: persistencia y validadores

- [x] Primera muestra paginada en Render (`7d62894`, [ejecución 43676778847](https://github.com/Baahl11/Soccer/actions/runs/43676778847/job/132698632979)): 96 eventos, sin errores. En la ventana real de 12: goles por equipo `1520972`; córners FT, amarillas FT/por equipo y roja sí/no `1570021`.
- [ ] Tiros/SOT: ocho candidatos revisados por familia sin positivo vigente ni válido al momento de captura en el primer corte. Quedan candidatos; cursor `23056`.
- [ ] Confirmar la respuesta HTTP y la visualización con una sesión OWNER autorizada. Un replay de validadores no sustituye esta prueba.


### Criterio de cierre del panel (incidente #217)
- [x] Recuperación observada de escritura: [tick canónico](https://github.com/Baahl11/Soccer/actions/runs/43937184150/job/133846227523), 6 eventos en respuesta, `database_persisted=true`; [replay independiente](https://github.com/Baahl11/Soccer/actions/runs/44003306297/job/133846808269), 5 eventos nuevos en `soccer_refresh_events`. Una ejecución no prueba estabilidad de memoria.
- [ ] Auditar en el mismo flujo la ficha reportada `1647367` y el control positivo de equipo `1545887`. El replay cubre siete familias, no el inventario completo de 16. Confirmar separado que las dos rutas API deniegan acceso anónimo.
- [ ] Con sesión OWNER autorizada, contrastar los bloques reales de la API y de la ficha para ambos IDs. No usar el token del workflow como sesión OWNER.
- [ ] Validar una nueva captura de SHOTS/SOT con ambos XI confirmados y timestamp verificable; conservar ausencias legítimas, sin rellenar el histórico.
- [ ] Confirmar escrituras programadas posteriores, sin cambiar guardia de memoria, cuotas ni puertas comerciales.
