# videos.ipynb — Pipeline de procesamiento de videos propios

## Por qué existe este notebook

El dataset WLASL tiene dos problemas concretos que afectan directamente el entrenamiento:

**Escasez de datos.** WLASL2000 tiene en promedio 10.5 videos por glosa. Para las 20 glosas que usamos en este trabajo, esa cifra es insuficiente para entrenar modelos que generalicen bien, especialmente en un problema de clasificación de alta dimensionalidad como reconocimiento de señas.

**Mantenimiento deficiente.** Los videos del dataset original provienen de fuentes externas (YouTube, sitios educativos). Con el tiempo, muchos links están caídos o los videos fueron eliminados. El script de descarga oficial (`video_downloader.py`) falla en una fracción significativa de los videos listados en `WLASL_v0.3.json`, lo que reduce aun más la cantidad de datos disponibles.

Para mitigar ambos problemas, filmamos nuestros propios videos de las 20 glosas que usamos en el trabajo. Este notebook procesa esos videos siguiendo el mismo pipeline del paper original, de modo que los datos propios sean completamente compatibles con el resto del dataset.

---

## Qué hace

### Paso 1 — `glosses_nuestras.json`

Escanea la carpeta `start_kit/videos_nuestros/<glosa>/` y construye un JSON con el mismo esquema que `glosses_valid.json` (el formato estándar del dataset).

Por cada video asigna:
- Un `video_id` secuencial (`N00001`, `N00002`, ...)
- Split train/val/test con ratio 4:1:1, garantizando al menos una muestra por split (igual que el paper, sección 4.3.2)
- Metadata leída con OpenCV: fps, resolución, `bbox` de frame completo
- La ruta al archivo como `url` para poder encontrarlo después

El archivo resultante se guarda en `data/glosses_nuestras.json`.

### Paso 2 — Extracción de keypoints

Por cada video en el JSON, lee el mp4 frame a frame y extrae keypoints con **MediaPipe Holistic**. El paper original usa OpenPose, que requiere una instalación C++ compleja y no está disponible vía pip. MediaPipe Holistic da el mismo output estructural:

- **Pose**: 33 landmarks de cuerpo → se mapean a los **25 joints de OpenPose BODY_25** usando una tabla de correspondencia explícita (Neck y MidHip se calculan como promedio de sus joints vecinos en MediaPipe)
- **Manos**: 21 landmarks por mano, idénticos en cantidad y orden a OpenPose

El resultado es un JSON por frame en `data/poses_nuestras/<video_id>/image_00001_keypoints.json`, con exactamente el mismo formato que los archivos en `data/pose_per_individual_videos/`. Esto garantiza que el código de entrenamiento puede leer ambas fuentes sin modificaciones.

La extracción es **idempotente**: si un video ya fue procesado (todos sus frames tienen JSON), se saltea.

---

## Cómo correr

1. Poner los videos en `start_kit/videos_nuestros/<glosa>/` (archivos `.mp4`)
2. Ejecutar el notebook de arriba hacia abajo
3. La primera corrida genera el JSON y los poses; las siguientes solo procesan videos nuevos

**Kernel**: usar el entorno del proyecto (`.venv`), que tiene `mediapipe` y `opencv` instalados.
