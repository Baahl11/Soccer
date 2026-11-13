# Soccer Edge — checklist de cierre del Expediente deportivo

**Alcance:** cada matchup elegible en el frontend OWNER/admin, reutilizando el registro canónico, PostgreSQL y los modelos existentes. Un partido con datos incompletos debe mostrar su estado y causa; «100 %» significa trazabilidad, no 100 % de partidos con pronóstico ni autorización de apuestas. SPORT FIRST, MARKET SECOND.

## 1. Base y despliegue
- [x] Reconciliar alcance del scanner y FE11 sin promover mercados (#189, #190).
- [x] Verificar un lote de 3 perfiles FE11 exportados e insertados en PostgreSQL (Actions 43432746878; segundo lote 43470612105).
- [x] Mostrar el Expediente deportivo en el frontend OWNER/admin (#192).
- [x] Filtrar por fuente y tiempo **antes** de elegir la observación prepartido más reciente (#193). Render verificado en Actions 43473070534.
- [x] Integrar lectura directa del RAW original en PostgreSQL, con identidad y digest, y tasas de gol por equipo sin presentarlas como xG ni como probabilidades por línea (#194, diez checks).
- [ ] Comprobar Render: commit esperado, HTML del expediente y acceso sin autorización rechazado. Registrar ejecución y SHA. El HTML no es prueba de sesión autenticada.

## 2. Prueba real PostgreSQL → match API → navegador
- [ ] Elegir el último lote FE11 con tres perfiles aún vigentes (NS, kickoff futuro, captura dentro de la vigencia). Registrar IDs del artefacto, instante de prueba y ACK. Si el lote venció, usar el siguiente; no re-fechar.
- [ ] Con sesión OWNER/admin autorizada, abrir cada ficha y comprobar coincidencia de fixture, local, visitante, kickoff, fuente y n. Evidencia requerida: 3 exportados → 3 registrados → 3 leídos → 3 visibles, o motivo individual verificable. No almacenar tokens, cookies ni HTML privado en logs públicos.
  - [x] `1647362` Argentina–Uruguay: captura OWNER confirma el perfil FE11, con los mismos equipos y timestamp `2026-11-13T17:14:03.578710+00:00` del artefacto del scanner 43470612105. Muestras por localía n=3 por lado, etiquetadas MUESTRA BAJA. Solo histórico; no RAW ni OOS.
  - [ ] `1640518` Colombia–Chile y `1647364` Perú–Bolivia: exportación/ACK confirmados en el mismo lote; visualización OWNER no verificada. Usar perfiles vigentes si estos caducan.
- [ ] Verificar un partido con RAW PostgreSQL y otro sin RAW. Probar enlace directo o recarga sin depender del listado del día. Confirmar que un usuario no OWNER no recibe los campos privados añadidos. Una prueba con datos de ejemplo no cierra este paso.

## 3. Completar las familias en esa misma ficha, sin recolectores paralelos
- [ ] Goles por equipo: localizar las salidas prepartido guardadas por lado y línea; no derivar un pronóstico nuevo ni reutilizar la calibración FT. Mantener λ separado de probabilidad.
- [ ] 1H/2H y córners: distinguir muestras descriptivas de proyecciones congeladas; mostrar ambos lados, fuentes, muestras, versión y bloqueos.
- [ ] Tarjetas: conectar solo salidas existentes que coincidan en fixture, ventana y tipo (amarillas, rojas, equipo o jugador); no mezclar puntos de tarjeta o reglas de liquidación.
- [ ] Props: verificar fixture, player_id, XI/minutos y procedencia prepartido; conectar tiros/SOT primero, luego demás familias cuando tengan evidencia. No reactivar backfills bloqueados por memoria sin prueba de capacidad.
- [ ] Para cada familia mostrar por separado evidencia, RAW, OOS y precio; distinguir NO VERIFICADO, muestra insuficiente, aplazado y diferido por cuota. Mantener muestra, fuente y hora al lado del dato.

## 4. Continuidad y aceptación
- [ ] Reconciliar por competición y familia cada fixture elegible (calculado, evidencia solamente, pendiente o bloqueado); medir diferencias entre detectado, importado, leído y mostrado, no solo éxito de Actions.
- [ ] Verificar refrescos y resultado oficial sin alterar la proyección congelada; evaluar OOS por familia y versión, sin extrapolar.
- [ ] Mantener BET comercial bloqueado mientras no pasen las puertas independientes de OOS, disponibilidad, precio ejecutable, liquidación y derechos. No se necesita abrir BET para dar por terminada la integración del expediente.

Marcar un punto solo con un resultado verificable y registrar fecha UTC, SHA y ejecución o prueba correspondiente.
