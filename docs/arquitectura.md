# Arquitectura de ImageEnhance AI

## Flujo

```
foto (archivo) ──► io_utils: BGR uint8 → float32 [0,1]
                     │
      ┌──────────────┴───────────────┐
  modo experimento               modo foto real
  noise: ruido con semilla       (sin referencia)
      │                              │
      ├──► classic: mediana / gaussiano / conv2d NumPy
      ├──► model: CNN residual (PyTorch, CPU)
      │
      ├──► metrics: PSNR y SSIM  ← solo si hay imagen limpia
      └──► analysis: histogramas, sigma ESTIMADO, Sobel  ← siempre
                     │
            classic: brillo/contraste/nitidez (opcional) ──► descarga PNG
```

## Módulos

| Archivo | Responsabilidad | Talleres |
|---|---|---|
| `imageenhance/io_utils.py` | Carga/guardado, BGR↔RGB, uint8↔float32 con `clip` | 01, 02 |
| `imageenhance/noise.py` | Ruido gaussiano y sal y pimienta reproducibles (semilla) | 04 |
| `imageenhance/classic.py` | `conv2d` propia en NumPy, kernels gaussiano/media, mediana, afín, realce | 01, 04 |
| `imageenhance/metrics.py` | PSNR y SSIM (validados contra scikit-image) | 04 |
| `imageenhance/analysis.py` | Histogramas, estadísticas, sigma estimado (Immerkær), Sobel | 02, 05 |
| `imageenhance/data.py` | Split por imagen, semillas por imagen/condición, parches | — |
| `imageenhance/model.py` | CNN residual, conteo de parámetros, inferencia por bloques | 11, 12 |
| `scripts/prepare_data.py` | Descarga BSDS500 con hashes y manifiesto del split | — |
| `scripts/compare_classic.py` | Ajusta mediana y gaussiano en validación | 04 |
| `scripts/train.py` | Entrenamiento reproducible, benchmark, reanudación | 11, 12 |
| `scripts/evaluate.py` | Evaluación final en prueba contra las líneas base | 04 |
| `scripts/kernel_experiment.py` | Kernel fijo contra kernel aprendido | 01, 04, 11 |
| `app.py` | Interfaz Streamlit | — |

## CNN

- 7 capas Conv2d 3×3 (3→32, 5×(32→32), 32→3), ReLU entre capas, sin BatchNorm, relleno `reflect`.
- Aprendizaje residual: `salida = entrada − red(entrada)`; la red estima el ruido.
- 48.003 parámetros: `(3·3·C_in + 1)·C_out` por capa (verificado en `tests/test_model.py`).
- Campo receptivo: 15×15 píxeles.
- Referencia de arquitectura: DnCNN (Zhang et al., 2017). Entrenada desde cero; sin pesos preentrenados.

## Decisiones

| Decisión | Motivo |
|---|---|
| Residual en vez de autoencoder | El autoencoder reduce resolución y pierde detalle fino; predecir el ruido converge más rápido |
| Sin BatchNorm | Lotes pequeños en CPU; menos piezas que explicar y depurar |
| Relleno `reflect` | Es el mismo borde que usa nuestra `conv2d` y OpenCV (reflect-101) |
| Ruido gaussiano con sigma aleatorio 5–50 | Un solo modelo para varios niveles de ruido (ciego al sigma) |
| Split oficial de BSDS500 por imagen | Ninguna foto de prueba aporta parches al entrenamiento |
| Filtros clásicos ajustados en validación | Comparación justa: ambos métodos eligen hiperparámetros sin ver la prueba |
| Entrenamiento en CPU | La Radeon integrada no tiene CUDA; la red es pequeña a propósito |
| Orden BGR en todo el pipeline | Es como carga OpenCV y como se entrenó la red; solo se convierte a RGB para mostrar |

## Limitaciones

- El ruido real de cámara depende de la señal (Poisson-Gaussiano) y no es exactamente gaussiano.
- La CNN no se entrenó para sal y pimienta; ahí la mediana debe ganar (se muestra en la evaluación).
- PSNR y SSIM solo son válidos con referencia limpia. En fotos reales solo hay una estimación del ruido.
- BSDS500 son fotos de 481×321; fotos de celular de alta resolución tienen otra escala de ruido y detalle.
