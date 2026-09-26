# Registro de experimentos

Todos los números de este documento salen de archivos en `results/` generados por los scripts.
Lo que todavía no se ha ejecutado aparece como **pendiente**.

## Datos

- **Dataset:** BSDS500 (Arbeláez et al., 2011), 500 fotos naturales a color de 481×321, uso académico.
- **Fuente:** espejo público `github.com/BIDS/BSDS500`, commit fijado en `data/splits.json` con SHA-256 por archivo.
- **Split oficial por fotografía:** train 200 · val 100 · test 200. Intersecciones: 0 (verificado en `tests/test_data.py`).
- **Ruido de entrenamiento:** gaussiano aditivo, sigma ~ U(5, 50) por parche, recortado a [0, 1], generado al vuelo con semilla `(2026, época)`.
- **Ruido de validación y prueba:** semilla `crc32(nombre|tipo|nivel) XOR 2026`, igual en cualquier máquina.

## Línea base clásica (validación, 100 fotos) — `results/classic_params.json`

Mejores parámetros por PSNR medio:

| Ruido | Sin filtro | Mediana | Gaussiano |
|---|---|---|---|
| Gaussiano σ=15 | 24.82 dB / 0.580 | 26.75 / 0.699 (3×3) | **27.90 / 0.777** (7×7, σ=0.8) |
| Gaussiano σ=25 | 20.53 dB / 0.399 | 24.82 / 0.575 (3×3) | **26.06 / 0.666** (7×7, σ=0.8) |
| Gaussiano σ=50 | 15.00 dB / 0.200 | 22.62 / 0.460 (7×7) | **23.19 / 0.551** (11×11, σ=1.5) |
| Sal y pimienta 5 % | 17.99 dB / 0.431 | **28.62 / 0.833** (3×3) | 24.76 / 0.631 (9×9, σ=1.1) |

Coincide con la Sesión 04: el gaussiano gana con ruido gaussiano y la mediana con ruido impulsivo.

## CNN

| Hiperparámetro | Valor (`configs/train.json`) |
|---|---|
| Parámetros | 48.003 |
| Parches | 64×64, 32 por foto y época (6.400 por época), giros y espejos |
| Lote | 32 |
| Épocas | 40 |
| Optimizador | Adam, lr 1e-3 → 1e-5 (coseno) |
| Pérdida | MSE |
| Validación | recorte central 160×160, σ=25 fijo |

- **Benchmark en el PC (Ryzen 5 5600GT, CPU):** pendiente.
- **Entrenamiento completo:** pendiente. Curvas en `results/denoise_cnn/curvas.png`.
- **Evaluación en prueba (200 fotos):** pendiente. Tabla en `results/test_summary.md`.
- **Kernel fijo contra aprendido:** pendiente. `results/kernel_experiment.png`.

## Prueba de humo (no es un resultado)

En el entorno de desarrollo, con 4 fotos y 160 pasos, el PSNR de validación pasó de 20.43 a 23.15 dB.
Solo demuestra que el pipeline aprende; no se usa como resultado del proyecto.
