# Revela

Software en Python que **analiza una fotografía, detecta qué problemas tiene y la mejora automáticamente**
con redes neuronales entrenadas por nosotros y procesamiento digital de imágenes clásico. Nada es generativo:
el sistema no inventa detalles que la foto no tenga.

Proyecto final de **Inteligencia Artificial II** — Institución Universitaria de Colombia.

Autores: Dylan Esteban Ricaurte Cuervo · Brayan Sneyder Garcia Camacho · Nicolas David Fontecha Poveda · Esteban Monroy

## Cómo funciona

1. **Diagnóstico:** se miden 11 características de la foto; un MLP entrenado por nosotros decide si hay ruido,
   desenfoque o compresión JPEG, y reglas sobre el histograma (umbrales elegidos en validación) deciden luz y contraste.
2. **Plan:** los pasos se ordenan siempre igual: ruido → detalle → contraste → luz.
3. **Restauración:** CNN de ruido y CNN de detalle.
4. **Retoque aprendido (FiveK):** una tercera CNN ajusta luz, contraste y color como lo haría un fotógrafo (casilla activa
   por defecto en fotos reales). Sin ella, niveles y gamma (reglas que vuelven a medir la foto antes de actuar).
5. **Color:** un detector con umbral no distingue la luz de la escena (atardecer, faroles) de un defecto (se equivocaba en
   33 de 100 fotos modernas limpias de DIV2K); el retoque aprendido sí lo resuelve porque aprendió de un fotógrafo.
   Si se detecta una posible dominante, la app avisa y ofrece además el balance de blancos clásico.
6. **Interfaz** (`app.py`; estilos en `imageenhance/ui.py` y colores de la universidad en `.streamlit/config.toml`):
   foto a la izquierda y panel a la derecha; antes/después con **deslizador**, lado a lado o **lupa ×3** de la zona que
   más cambió; los pasos aplicados con su origen (nuestra IA, clásico o externo) y, en el modo experimento, PSNR y SSIM
   contra la foto limpia.

## Modelos

| Modelo | Tarea | Origen | Parámetros |
|---|---|---|---|
| `models/denoise_cnn_v2.pt` | Quitar ruido (gaussiano y de celular) | **Entrenado por nosotros** (DIV2K, 10 capas) | 75.747 |
| `models/detail_cnn_v3.pt` | Desenfoque, JPEG y baja resolución | **Entrenado por nosotros** (DIV2K, 16 capas, He init) | 293.619 |
| `models/retouch_fivek.pt` | Retoque: luz, contraste y color | **Entrenado por nosotros** (MIT-Adobe FiveK, experto C; predice 13 ajustes globales) | 65.517 |
| `models/diagnosis_mlp.pt` | Diagnosticar ruido / desenfoque / JPEG | **Entrenado por nosotros** (11 → 32 → 16 → 3) | 963 |
| `models/external/realesr-general-x4v3.pth` | Superresolución (solo manual) | **Externo:** Real-ESRGAN (Wang et al., 2021), BSD-3, generativo | 1.213.296 |

Las CNN son residuales al estilo DnCNN (Zhang et al., 2017): estiman lo que sobra y lo restan. Las versiones
anteriores (`denoise_cnn.pt`, `detail_cnn.pt`, `detail_cnn_v2.pt`) se conservan para reproducir las comparaciones.

## Resultados (200 fotos de prueba de BSDS500, nunca vistas al entrenar; PSNR dB / SSIM)

**Ruido:**

| Ruido | Sin filtrar | Mediana | Gaussiano | CNN v1 | **CNN v2** |
|---|---|---|---|---|---|
| Gaussiano σ = 15 | 24,83 | 26,96 | 28,04 | 30,94 | **31,68 / 0,877** |
| Gaussiano σ = 25 | 20,54 | 24,96 | 26,17 | **29,62** | 29,11 / 0,812 |
| Gaussiano σ = 50 | 15,01 | 22,61 | 23,23 | 25,04 | **25,20 / 0,644** |
| Sal y pimienta 5 % | 17,97 | **28,88** | 24,84 | 21,80 | 19,51 |
| Celular (ruido de cámara) | 29,09 | 26,57 | 27,08 | 29,15 | **30,97 / 0,879** |

**Detalle:**

| Condición | Sin restaurar | Nitidez clásica | CNN v2 | **CNN v3** | Real-ESRGAN |
|---|---|---|---|---|---|
| Desenfoque σ 1,5 | 25,79 | 26,99 | **30,59** | 27,62 | 26,33 |
| JPEG calidad 20 | 28,21 | 27,48 | **28,34** | 28,13 | 26,45 |
| Muy borrosa (σ 3 → ×2 → JPEG 60) | 22,94 | 23,09 | 23,48 | **24,22** | 23,90 |
| Celular (cámara → JPEG 85) | 28,80 | 27,61 | 28,64 | **29,45** | 25,95 |

**Retoque aprendido** (498 fotos de prueba de FiveK completas, contra el retoque del experto C): sin tocar 17,91 dB →
nuestras reglas 19,90 → **retoque aprendido 24,47 dB / 0,870** (mejora 449 de 498 fotos).

**Diagnóstico** (800 muestras de prueba): los 3 problemas correctos a la vez en **81,6 %** (MLP) contra 64,0 % (mejor
umbral sobre una sola característica).

**Sistema automático completo** (9 condiciones; `results/system_eval/bsds500_denoise_cnn_v2_detail_cnn_v3.md`):
ruido σ 15 24,83 → **31,16** dB (mejora 194/200); foto oscura y ruidosa 12,97 → **21,93**; celular 28,80 → **29,70**
(con métodos clásicos: 27,13); 167/200 fotos limpias quedan intactas. Modelos y umbrales se eligen en las 100 fotos
de validación (`scripts/evaluate_system.py --split val`) y la prueba solo confirma.

## Instalación (Windows, PowerShell)

```powershell
git clone https://github.com/DylanRC4/Proyecto-final---Mejora-de-Imagen-con-IA.git
cd Proyecto-final---Mejora-de-Imagen-con-IA
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py    # opcional: Real-ESRGAN (solo en Ajuste manual)
.\.venv\Scripts\python.exe scripts\prepare_data.py       # opcional: fotos BSDS500 de ejemplo
```

En Linux, instalar primero PyTorch para CPU (evita descargar CUDA) y usar `.venv/bin/python`:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

## Uso

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si se cambia el código, hay que cerrar Streamlit y volver a abrirlo (los módulos quedan en memoria).

## Reproducir los experimentos

```powershell
python -m pytest -q                                           # 64 pruebas
python scripts\prepare_div2k.py                               # DIV2K reducido ×2 (700/100/100)
python scripts\train.py --config configs\train_noise_v2.json --name denoise_cnn_v2   # CNN de ruido v2
python scripts\train.py --config configs\train_detail_v3.json --name detail_cnn_v3  # CNN de detalle v3
python scripts\train_diagnosis.py                             # MLP de diagnóstico
python scripts\evaluate.py; python scripts\evaluate_detail.py # evaluación de cada red
python scripts\evaluate_system.py [--split val]               # sistema completo
python scripts\prepare_fivek.py; python scripts\train_retouch.py --epochs 40   # retoque (FiveK 480p en data/raw/fivek)
```

Todo usa semillas fijas y splits por fotografía (ninguna foto aparece en dos subconjuntos).

## Limitaciones

- Entrenamos con degradaciones sintéticas; no medimos con un dataset de ruido real con referencia (PolyU, SIDD).
- El retoque aprendió el gusto de UNA persona (experto C) sobre fotos de cámara réflex; en fotos de celular ya procesadas
  puede exagerar, por eso es una casilla que se puede apagar. FiveK es solo para investigación.
- Con grano grueso (3-6 px), con sal y pimienta y con desenfoque suave sin compresión (v3) el rendimiento baja.
- Con todas las degradaciones juntas, el plan con métodos clásicos queda levemente mejor (18,21 contra 18,05 dB).
- No recupera caras de pocos píxeles ni fotos muy movidas: eso solo lo hace una IA generativa, que no usamos.

## Créditos

- BSDS500: Arbeláez et al., 2011 (uso académico). DIV2K: Agustsson y Timofte, 2017.
- MIT-Adobe FiveK: Bychkovsky et al., 2011 (solo investigación); versión 480p y split de prueba del paper 3D-LUT (Zeng et al., 2020).
- Real-ESRGAN: Wang et al., 2021, `github.com/xinntao/Real-ESRGAN` (BSD-3). DnCNN: Zhang et al., 2017.
