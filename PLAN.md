# Plan de desarrollo

## Enfoque del trabajo

El trabajo se plantea como un estudio exploratorio y reproducible sobre reconocimiento de señas aisladas usando un subconjunto controlado de WLASL. Por restricciones de datos disponibles y poder de computo, el alcance principal se limita a las 20 glosas con mas instancias validas.

La comparacion central debe ser justa: correr el modelo base bajo las mismas condiciones de datos disponibles y comparar contra nuestras variantes. Las mejoras propuestas incluyen limpieza/auditoria de datos, datos propios, augmentation, features de pose/boca y fusion multimodal.

No se afirma resolver WLASL completo ni reconocimiento abierto de lengua de señas. La contribucion esta en construir un pipeline controlado, analizar sus limitaciones y evaluar si las mejoras son extrapolables.

## Principios de evaluacion

- Usar el mismo dataset final para baseline y variantes.
- Reportar accuracy, macro-F1, matriz de confusion y resultados por glosa.
- Documentar cantidad de videos por glosa, split y fuente.
- Separar claramente datos WLASL y videos propios.
- Presentar MediaPipe como compatible en formato con OpenPose, no como equivalente perfecto.
- Usar clustering como auditoria exploratoria, no como detector definitivo de etiquetas incorrectas.
- Incluir al menos una evaluacion estricta por signer o fuente si la metadata lo permite.

## Sprint 1 - Protocolo y dataset auditable

### Objetivo

Cerrar el protocolo experimental y dejar el dataset final claramente definido, reproducible y auditable.

### Tareas

- Confirmar que la descarga y el pipeline usan `glosses_valid.json` y no `WLASL_v0.3.json` completo.
- Documentar el criterio de seleccion de top 20 glosas.
- Sincronizar `glosses_valid.json` con los videos realmente descargados.
- Generar un resumen del dataset final:
  - videos por glosa
  - videos por split
  - videos por fuente (`wlasl` / `own`)
  - videos faltantes o descartados
- Definir metadata minima para videos propios:
  - `source`
  - `signer_id`
  - `recording_session`
  - `gloss`
  - `split`

### Entregables

- Dataset top 20 validado.
- Tabla/resumen del dataset final.
- Criterio de seleccion y filtrado documentado.

### Definition of done

- El baseline y todas las variantes pueden leer el mismo dataset final.
- Existe una tabla clara con cantidades por glosa, split y fuente.
- Se puede explicar por que el trabajo usa 20 glosas sin prometer generalizacion total.

## Sprint 2 - Baselines reproducibles

### Objetivo

Obtener resultados base confiables antes de introducir mejoras propias.

### Tareas

- Correr el modelo base I3D sobre el dataset top 20.
- Correr el modelo base con augmentation.
- Correr TGCN si el pipeline esta operativo.
- Guardar resultados, curvas de entrenamiento y configuraciones usadas.
- Generar metricas:
  - accuracy
  - macro-F1
  - matriz de confusion
  - metricas por glosa

### Entregables

- Resultados baseline I3D.
- Resultados baseline con augmentation.
- Resultados TGCN si aplica.
- Primera tabla comparativa.

### Definition of done

- Hay un numero base defendible contra el cual comparar.
- Los experimentos son reproducibles desde configuraciones o notebooks.
- Se sabe que glosas confunde mas el modelo base.

## Sprint 3 - Integracion de videos propios

### Objetivo

Evaluar si los videos propios mejoran el rendimiento y bajo que condiciones.

### Tareas

- Mantener `glosses_nuestras.json` separado del dataset WLASL original.
- Generar un dataset combinado solo para experimentos.
- Asegurar que cada video propio tenga `signer_id` y `source = own`.
- Correr experimentos:
  - WLASL only
  - WLASL + propios
  - WLASL + augmentation
  - WLASL + propios + augmentation
- Comparar resultados contra baseline.
- Analizar si la mejora aparece en todas las glosas o solo en algunas.

### Entregables

- JSON combinado para experimentos.
- Tabla de ablations con y sin videos propios.
- Discusion preliminar sobre domain shift.

### Definition of done

- Se puede responder si los videos propios ayudan o no.
- La comparacion no mezcla cambios de datos con cambios de arquitectura sin control.
- Queda documentado el riesgo de domain shift.

## Sprint 4 - Evaluacion estricta de generalizacion

### Objetivo

Medir si el modelo generaliza a signers o fuentes no vistas, especialmente con videos propios.

### Tareas

- Implementar split por `signer_id` para videos propios si la metadata esta disponible.
- Implementar split por fuente o sesion como alternativa si no alcanza la cantidad de signers.
- Evaluar al menos una configuracion estricta:
  - train sin un signer propio y test en ese signer
  - train en WLASL + algunos propios y test en propios no vistos
  - train en WLASL + propios y test separado solo WLASL
- Comparar split aleatorio vs split agrupado.

### Entregables

- Resultados de generalizacion estricta.
- Tabla comparando split normal y split por grupo.
- Discusion sobre cuanto baja o mejora el rendimiento.

### Definition of done

- El informe puede distinguir performance en condiciones faciles y estrictas.
- Se evita afirmar generalizacion si el resultado solo vale para split aleatorio.
- Existe evidencia para discutir si el modelo aprende señas o tambien condiciones de grabacion.

## Sprint 5 - Auditoria de calidad con clustering

### Objetivo

Usar clustering/outlier detection como herramienta exploratoria para encontrar posibles videos problematicos.

### Tareas

- Ejecutar el notebook/pipeline de clustering sobre las glosas usadas.
- Generar ranking de videos sospechosos.
- Revisar manualmente una muestra de outliers.
- Clasificar hallazgos:
  - posible etiqueta incorrecta
  - baja calidad de pose
  - signer o dialecto atipico
  - video ambiguo
- Decidir si se excluyen videos y documentar el criterio.

### Entregables

- `mislabels_sospechosos.csv` o equivalente.
- Ejemplos revisados manualmente.
- Criterio de exclusion o decision de mantener datos.

### Definition of done

- El clustering se presenta como apoyo para auditoria, no como verdad automatica.
- Hay evidencia visual o manual para los casos mas importantes.
- Si se excluyen videos, el cambio queda justificado.

## Sprint 6 - Fusion multimodal

### Objetivo

Evaluar si combinar modalidades mejora sobre modelos unimodales sin caer en sobreajuste no detectado.

### Tareas

- Evaluar modelos unimodales:
  - I3D
  - TGCN
  - features de boca si estan disponibles
- Evaluar fusiones:
  - I3D + TGCN
  - I3D + TGCN + boca
  - concatenacion simple
  - proyeccion/suma
  - bilinear fusion si el dataset lo soporta
- Comparar contra los baselines del Sprint 2.
- Revisar macro-F1 y matriz de confusion, no solo accuracy.

### Entregables

- Tabla comparativa unimodal vs multimodal.
- Analisis de sobreajuste.
- Seleccion del mejor modelo final.

### Definition of done

- La fusion se justifica solo si mejora consistentemente.
- Si no mejora, queda documentado como resultado valido.
- Se puede explicar que modalidad aporta y cual no.

## Sprint 7 - Informe final y defensa

### Objetivo

Convertir los experimentos en una narrativa clara, honesta y defendible.

### Tareas

- Escribir la motivacion:
  - datos faltantes
  - restricciones de computo
  - ruido de etiquetas
  - necesidad de un subconjunto controlado
- Describir el pipeline completo.
- Presentar resultados por etapas:
  - baseline
  - augmentation
  - videos propios
  - evaluacion estricta
  - clustering
  - fusion multimodal
- Incluir limitaciones:
  - top 20 no representa WLASL completo
  - domain shift de videos propios
  - MediaPipe no equivale perfectamente a OpenPose
  - clustering no confirma mislabels automaticamente
  - pocos datos para fusion multimodal
- Preparar respuestas a preguntas criticas.

### Entregables

- Informe final.
- Tablas y figuras definitivas.
- Seccion de limitaciones.
- Guion breve de defensa.

### Definition of done

- El trabajo no promete mas de lo que mide.
- Las mejoras se comparan contra un baseline justo.
- Las limitaciones aparecen como parte del aporte, no como debilidades escondidas.

## Riesgos principales

- La cantidad de datos puede ser insuficiente para fusion multimodal compleja.
- Los videos propios pueden mejorar accuracy sin mejorar generalizacion.
- El split aleatorio puede inflar resultados si hay signers/fuentes repetidas.
- MediaPipe puede producir keypoints con errores distintos a OpenPose.
- La descarga incompleta de WLASL puede dificultar comparacion directa con el paper original.

## Prioridad si falta tiempo

Si el tiempo queda corto, priorizar:

1. Dataset auditable.
2. Baseline reproducible.
3. Ablation con videos propios y augmentation.
4. Evaluacion estricta por signer/fuente.
5. Metricas completas y matriz de confusion.

La fusion multimodal y el clustering son valiosos, pero deben quedar despues de tener una comparacion base solida.
