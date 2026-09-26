# ImageEnhance AI

Software en Python que **analiza una fotografía, detecta qué problemas tiene y la mejora automáticamente**
combinando inteligencia artificial entrenada por nosotros, un modelo externo de superresolución y
procesamiento digital de imágenes clásico.

Proyecto final de **Inteligencia Artificial II** — Institución Universitaria de Colombia.

Autores:

- Dylan Esteban Ricaurte Cuervo
- Brayan Sneyder Garcia Camacho
- Nicolas David Fontecha Poveda

## Características principales

- **Mejora automática:** un clasificador MLP diagnostica la foto (ruido, desenfoque, compresión JPEG) y reglas
  sobre el histograma detectan iluminación, contraste y color. Con ese diagnóstico el programa propone los
  pasos y los aplica en el orden correcto: ruido → detalle → luz → contraste → color.
- **Reducción de ruido** con una CNN residual entrenada desde cero por nosotros.
- **Restauración de detalle** (desenfoque leve y artefactos de compresión) con Real-ESRGAN, con intensidad ajustable.
- **Aumento de resolución ×2 y ×4** con Real-ESRGAN.
- **Iluminación** (corrección gamma), **contraste** (niveles automáticos y CLAHE), **color** (balance de blancos)
  y **nitidez** (máscara de desenfoque).
- **Herramientas individuales** para aplicar cada mejora por separado.
- **Comparación antes/después** y **descarga** del resultado en PNG.
- **Métricas honestas:** PSNR y SSIM solo cuando existe la imagen limpia de referencia (modo experimento).
  En fotos reales se compara a la vista.
- **Análisis de la imagen:** histogramas por canal, estadísticas, ruido estimado y gradiente de Sobel.
- **Pensado para un portátil:** todo corre en CPU, las fotos grandes se procesan por bloques y, una vez
  instalado, no necesita internet.

## Modelos

| Modelo | Tarea | Origen | Parámetros | Resultado medido |
|---|---|---|---|---|
| `models/denoise_cnn.pt` | Quitar ruido | **Entrenado por nosotros** (40 épocas, CPU, ~36 min) | 48.003 | Ver tabla de ruido |
| `models/diagnosis_mlp.pt` | Diagnosticar ruido / desenfoque / JPEG | **Entrenado por nosotros** | 963 | 83,5 % de fotos con los 3 diagnósticos correctos |
| `models/external/realesr-general-x4v3.pth` | Detalle y superresolución | **Externo preentrenado:** Real-ESRGAN (Wang et al., 2021), licencia BSD-3 | 1.213.296 | Se evalúa a la vista (ver limitaciones) |

La arquitectura de la CNN de ruido se inspira en DnCNN (Zhang et al., 2017). Sus pesos y los de la MLP salen
de nuestro entrenamiento; no se usan pesos preentrenados de terceros para ellos.

## Resultados

**Reducción de ruido** en las 200 fotos de prueba de BSDS500, que ningún modelo vio al entrenar
(PSNR en dB / SSIM, más alto es mejor):

| Ruido | Sin filtrar | Mediana | Gaussiano | Nuestra CNN |
|---|---|---|---|---|
| Gaussiano σ = 15 | 24,83 / 0,563 | 26,96 / 0,714 | 28,04 / 0,785 | **30,94 / 0,872** |
| Gaussiano σ = 25 | 20,54 / 0,385 | 24,96 / 0,584 | 26,17 / 0,669 | **29,62 / 0,837** |
| Gaussiano σ = 50 | 15,01 / 0,193 | 22,61 / 0,481 | 23,23 / 0,567 | **25,04 / 0,633** |
| Sal y pimienta 5 % | 17,97 / 0,419 | **28,88 / 0,858** | 24,84 / 0,645 | 21,80 / 0,516 |

La CNN gana con ruido gaussiano (supera al gaussiano en 200 de 200 fotos con σ = 25). Con sal y pimienta gana
la mediana: la red nunca vio ese ruido al entrenar.

**Diagnóstico automático** en 800 fotos de prueba degradadas:

| Problema | MLP (nuestra) | Mejor umbral simple |
|---|---|---|
| Ruido | **88,4 %** | 78,7 % |
| Desenfoque | **94,9 %** | 78,7 % |
| Compresión JPEG | **100 %** | 99,8 % |
| Los 3 correctos a la vez | **83,5 %** | 60,4 % |

Detalles, curvas y ejemplos en `results/` y `docs/experimentos.md`.

## Instalación (Windows, PowerShell)

```powershell
git clone https://github.com/DylanRC4/Proyecto-final---Mejora-de-Imagen-con-IA.git
cd Proyecto-final---Mejora-de-Imagen-con-IA
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py    # Real-ESRGAN (4,9 MB, verifica SHA-256)
.\.venv\Scripts\python.exe scripts\prepare_data.py       # opcional: fotos BSDS500 de ejemplo
```

## Uso

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Se abre en el navegador (`http://localhost:8501`). Carga una foto y usa **Mejora automática** o **Herramientas**.

## Reproducir los experimentos

```powershell
python -m pytest -q                       # pruebas
python scripts\compare_classic.py        # ajusta mediana y gaussiano en validación
python scripts\train.py                  # entrena la CNN de ruido (usa --bench 30 para medir velocidad)
python scripts\evaluate.py               # evalúa en prueba contra los filtros clásicos
python scripts\kernel_experiment.py      # kernel fijo contra kernel aprendido
python scripts\train_diagnosis.py        # entrena y evalúa la MLP de diagnóstico
```

Todo usa semillas fijas y el split oficial de BSDS500 por fotografía (200 entrenamiento / 100 validación /
200 prueba), así que los resultados se pueden repetir.

## Estructura

```
imageenhance/   io_utils · noise · classic · enhance · metrics · analysis · data
                model (CNN propia) · diagnosis (MLP propia) · external (Real-ESRGAN) · pipeline (modo automático)
scripts/        prepare_data · compare_classic · train · evaluate · kernel_experiment · train_diagnosis · download_models
models/         pesos entrenados por nosotros (+ external/, descargado, fuera de Git)
results/        métricas, curvas y ejemplos
docs/           arquitectura y registro de experimentos
tests/          pruebas con pytest
app.py          interfaz Streamlit
```

## Limitaciones

- El ruido real de una cámara depende del brillo y no es exactamente gaussiano: en fotos reales la mejora es menor
  que en los experimentos.
- Real-ESRGAN inventa texturas creíbles: se ve más nítido pero puede dar un aspecto "pintado" y obtiene menos PSNR
  que una ampliación bicúbica. Por eso su intensidad es ajustable y se evalúa a la vista.
- Sin referencia no se puede distinguir un atardecer de una foto amarillenta: la corrección automática de color es conservadora.
- No recupera desenfoques fuertes, fotos muy movidas ni rostros de muy pocos píxeles.

## Créditos

- BSDS500: Arbeláez et al., 2011 (uso académico), descargado del espejo `github.com/BIDS/BSDS500`.
- Real-ESRGAN: Wang et al., 2021, `github.com/xinntao/Real-ESRGAN` (BSD-3).
- DnCNN: Zhang et al., 2017, referencia de arquitectura.
