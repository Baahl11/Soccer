# Soccer · recuperación de datos OOS por mercado

Corte de evidencia: 2026-11-17 UTC. El alcance es acumular pares verificables para investigación, no activar apuestas. Una tarjeta visible, una tasa λ, una corrida exitosa y un registro CLV no son un resultado OOS.

| Familia | Evidencia actual | Recuperación y prueba de cierre |
| --- | --- | --- |
| 1X2 FT · BTTS · totales FT | El piloto FT reportó 622 registros congelados y 4 nuevos partidos calificados sobre 45 previos. Es un recibo por partido, no un conteo independiente por mercado o línea. | Reutilizar el cruce FT programado. Publicar pares y altas por familia, línea y versión. |
| Goles por equipo | Se comprobaron proyecciones guardadas en la ficha; eso no certifica el cruce de cada selección. | Reutilizar marcador oficial FT local/visitante; reconciliar las predicciones congeladas por lado y línea. |
| 1H · 2H | No hay prueba reciente de acumulación por periodo en este corte. | Recuperar marcadores oficiales HT/FT en tiempo reglamentario; validar 2H FT−HT. Separar pronóstico 2H prepartido del pronóstico al descanso. |
| Córners FT · córners por equipo | El inventario semanal acotado del 17 nov recuperó un resultado adicional desde observaciones, pero cero pares formación prepartido–resultado; varias semanas truncadas. No mide todos los pares de la línea base. | Reusar captura final canónica; cruzar córners oficiales por lado con predicción prepartido ya guardada. Contar base y challenger por separado. |
| Amarillas FT · amarillas por equipo · roja sí/no | Cinco tarjetas de equipo verificadas en el panel; no hay un recibo reciente de acumulación OOS por estas tres familias. | Recuperar eventos/estadísticas finales con fuente verificable, lado y semántica de liquidación. No combinar amarillas, rojas, puntos ni segunda amarilla sin reglas explícitas. |
| Tiros · tiros a puerta | Cruce estricto publicado: 63 y 61 filas jugador-partido, respectivamente; 0 nuevas, 7 exclusiones por timestamp de registro ausente/inválido por familia. Replay reciente: nuevos eventos con XI sin confirmar; ningún positivo. | Reutilizar captura oficial de jugador y cruce estricto. Separar XI no confirmado, falta de registro as-of y resultado final no verificado. No contar jugador ausente como cero. |
| Otras familias de jugador o mercados del panel | Sin prueba de cobertura en las ejecuciones citadas. No asumir que todos los bloques del panel representan mercados congelados. | Contrastar catálogo exacto con productor, almacenamiento, observación oficial y validador dedicado. Identificar las familias sin ruta antes de reportarlas activas. |

## Contrato de recuperación

1. Reutilizar scheduler, observaciones finales, almacén congelado y cuotas existentes. Recuperar resultados históricos solo contra una predicción genuina anterior al punto de corte del mercado; si no existe, arrancar acumulación prospectiva, sin reconstruir la predicción.
2. Unir por identidad canónica, mercado/periodo, lado o jugador, línea, versión, punto de corte y fuente. Contar congelados elegibles, finales verificados, pares únicos, altas y exclusiones por causa. Mantener separados partidos, jugadores y selecciones por línea.
3. Publicar evidencia de primera alta y deltas posteriores por familia. Cero en un corte acotado no equivale a ausencia global. No elevar cap, cambiar permisos, mezclar modelos ni mover las puertas OOS/CLV/BET.

## Evidencia

- [Recibo oficial FT](https://github.com/Baahl11/Soccer/actions/runs/44084161282/job/134042166552)
- [Cruce oficial SHOTS/SOT](https://github.com/Baahl11/Soccer/actions/runs/44063313216/job/133970571704)
- [Inventario semanal de córners](https://github.com/Baahl11/Soccer/actions/runs/44031807777/job/133865368004)
- [Corrección de la cola de observaciones finales](https://github.com/Baahl11/Soccer/pull/223)
- [Último replay de captura de jugador](https://github.com/Baahl11/Soccer/actions/runs/44031807777/job/134172470848)
