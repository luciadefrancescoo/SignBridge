# clustering.ipynb — Detección de videos mal etiquetados por clustering

## Por qué existe este notebook

Al revisar manualmente los videos de las 20 glosas que usamos en el trabajo, encontramos videos claramente mal etiquetados: el video existe, pero la seña que muestra no corresponde a la etiqueta que tiene en el JSON del dataset.

Esto levantó una pregunta más amplia: si hay errores en las 20 glosas que revisamos, ¿cuántos errores hay en las 2000 glosas del dataset completo? WLASL fue construido de forma semi-automática a partir de videos de YouTube y sitios educativos, con anotaciones manuales parciales. El paper original reconoce variación entre signers y ambigüedad lingüística, pero no reporta una revisión sistemática de la calidad de las etiquetas.

Este notebook hace esa revisión de forma automática: dado que todos los videos ya tienen poses pre-computadas en `data/pose_per_individual_videos/`, podemos extraer una representación numérica de cada seña y detectar qué videos son anómalos dentro de su glosa.

---

## Qué hace

### Representación de cada video (features)

Para cada video se computa un vector de **220 dimensiones**:

1. Se cargan todos los frames del video (JSONs de poses)
2. En cada frame se extraen los **55 keypoints relevantes**: 13 joints de upper body + 21 de mano izquierda + 21 de mano derecha (igual que el paper, sección 4.2)
3. Los keypoints se **normalizan**: se centra en el cuello (joint 1 de BODY_25) y se escala por la distancia cuello→cadera. Esto hace la representación invariante a la resolución del video y a la posición del signer en el cuadro
4. Se calcula **media y desvío estándar** de cada coordenada a lo largo del video → vector de 55 joints × 2 coords × 2 estadísticos = 220 dims

La primera corrida tarda ~15 minutos (21K videos, lectura paralela con 8 threads). El resultado se cachea en `data/features_cache.npy` y las corridas siguientes son instantáneas.

### Detección de mislabels (within-gloss outlier score)

El método se basa en una premisa simple: **los videos de una misma glosa deberían tener features similares**. Un video cuya seña es diferente a las otras de su glosa — ya sea porque está mal etiquetado o porque es de baja calidad — va a estar lejos del centroide de su grupo.

Para cada glosa:
1. Se calcula el **centroide** (media de features) de todos sus videos
2. Se calcula la **distancia euclidiana** de cada video al centroide
3. Se **z-normaliza** esa distancia dentro de la glosa

Un video con z-score > 2.5 está a más de 2.5 desvíos estándar del centro de su glosa y se marca como sospechoso.

Se eligió detección within-gloss en lugar de clustering global (k-means con k=2000) porque con un promedio de ~10 videos por glosa los clusters globales son inestables. El enfoque within-gloss responde directamente la pregunta de interés: ¿este video se parece a los otros de su misma glosa?

### Outputs

| Archivo | Descripción |
|---|---|
| `data/features_cache.npy` | Features de los 21K videos (cache) |
| `data/mislabels_sospechosos.csv` | Tabla rankeada de videos sospechosos con video_id, glosa, z-score, split y fuente |
| `data/clustering_overview.png` | Scatter PCA 2D con sospechosos en rojo + histograma de scores |
| `data/tsne_outliers.png` | Idem con t-SNE (mejor separación, más lento) |
| `data/gloss_contamination.png` | Distribución de la "tasa de contaminación" por glosa |

La función `inspect_gloss("nombre")` permite explorar cualquier glosa individual: muestra su posición en el espacio global y un bar chart de los scores de cada uno de sus videos.

---

## Limitaciones

El método detecta videos **atípicos dentro de su glosa**, no mislabels con certeza. Un video puede tener z-score alto por otras razones: baja calidad de video (sin detección de pose), signer con estilo muy diferente al resto, o dialecto regional. El CSV es un punto de partida para revisión manual, no una lista definitiva de errores.
