# Guía de sustentación — ImageEnhance AI

Misma estructura que la guía de los talleres: qué se hizo, datos de memoria, preguntas probables y una
frase para cerrar. Todos los números salen de `results/` y se pueden regenerar con los scripts.

## En 30 segundos

Construimos un software que quita ruido de fotografías. Primero con filtros clásicos diseñados a mano
(mediana y gaussiano, Sesión 04) y luego con una red convolucional pequeña que **entrenamos nosotros
desde cero** con pares de fotos limpias y ruidosas. Medimos todo con PSNR y SSIM en 200 fotos que la
red nunca vio, y la interfaz en Streamlit permite repetir el experimento con cualquier foto.

## Datos que debes tener de memoria

- **Dataset:** BSDS500, 500 fotos de 481×321. Split oficial **por fotografía**: 200 entrenamiento,
  100 validación, 200 prueba. Intersección 0. Commit del espejo y SHA-256 de cada foto en `data/splits.json`.
- **Ruido:** gaussiano aditivo. Entrenamiento con σ aleatorio entre **5 y 50** (escala 0-255); validación σ = 25.
  Semilla de cada foto de prueba: `crc32(nombre|tipo|nivel) XOR 2026`.
- **Red:** 7 convoluciones 3×3, 32 filtros, ReLU, **48.003 parámetros**.
  Primera capa: (3·3·3 + 1)·32 = 896. Intermedias: (3·3·32 + 1)·32 = 9.248 (×5). Última: (3·3·32 + 1)·3 = 867.
- **Residual:** la red predice el ruido y lo resta: `limpia = ruidosa − red(ruidosa)`.
- **Entrenamiento:** {{TRAIN_SUMMARY}}
- **Resultado en prueba (PSNR / SSIM medios, 200 fotos):**

{{TEST_TABLE}}

- **Kernel aprendido:** {{KERNEL_SUMMARY}}

## Preguntas probables

| Pregunta | Cómo responder |
|---|---|
| **¿Usaron un modelo preentrenado?** | No. DnCNN (Zhang et al., 2017) es la referencia de la arquitectura, pero los pesos salen de nuestro entrenamiento: está el log por época, la configuración, la semilla y el historial de commits. |
| **¿Por qué una CNN si el gaussiano ya funciona?** | El gaussiano aplica el mismo kernel en toda la imagen: suaviza igual un cielo que un borde. La CNN combina 32 filtros por capa con ReLU y decide según el contexto local. El número que lo respalda es la tabla de prueba. |
| **¿Qué diferencia hay entre tu kernel de la Sesión 01 y una capa Conv2d?** | Es la misma cuenta: multiplicar posición contra posición y sumar. Hay un test que lo demuestra (`test_conv2d_numpy_es_la_misma_operacion_que_nn_conv2d`). La diferencia es quién elige los pesos: en la Sesión 01 los escribí yo; en la CNN los elige el descenso de gradiente. |
| **¿Por qué aprendizaje residual?** | Es más fácil aprender el ruido (pequeño, de media cero) que dibujar toda la imagen. Si la red estima ruido cero, la salida es exactamente la entrada (lo comprueba un test), no una imagen gris. |
| **¿Por qué ReLU y no sigmoide?** | La sigmoide se satura y su pendiente se hace casi cero en capas profundas. La ReLU deja pasar el gradiente completo en la parte positiva. La última capa no lleva activación porque el ruido puede ser negativo. |
| **¿Una capa lineal no bastaba?** | Un filtro lineal es un promedio ponderado fijo, igual que el perceptrón traza una sola recta (Sesión 11). {{KERNEL_ANSWER}} Las capas con ReLU curvan la frontera, como la MLP de la Sesión 12. |
| **¿Por qué MSE como pérdida?** | Porque PSNR = 10·log10(1/MSE): minimizar el MSE es maximizar el PSNR. |
| **¿Cómo evitaron fuga de datos?** | Split por fotografía; los parámetros de mediana y gaussiano se eligieron en validación; la prueba se usó una sola vez al final. Ningún parche de una foto de prueba se usó para entrenar. |
| **¿Por qué no muestran PSNR en una foto real?** | Porque PSNR y SSIM comparan contra la imagen limpia, y en una foto real no existe. La app solo muestra un sigma **estimado** (Immerkær), y lo dice. |
| **¿Cómo estiman el ruido sin referencia?** | Con un kernel 3×3 cuyos coeficientes suman 0: anula zonas planas y rampas, y lo que queda es mayormente ruido. Es otra convolución, como las de la Sesión 04. |
| **¿Por qué la CNN pierde con sal y pimienta?** | Nunca vio ese ruido: se entrenó solo con gaussiano. La mediana gana porque ordena y descarta los extremos (Sesión 04). Lo dejamos en la evaluación a propósito, para no esconder el límite. |
| **¿Por qué no usaron la GPU?** | La Radeon integrada no tiene CUDA. Por eso la red es pequeña: 48 mil parámetros, ~0,25 s por lote de 32 parches en CPU. |
| **¿Hay sobreajuste?** | Cada época usa parches nuevos, con giros y espejos, y se guarda la época con mejor PSNR de validación. La curva de `results/denoise_cnn/curvas.png` muestra si la validación deja de subir. |
| **¿Por qué BGR en todo el pipeline?** | OpenCV carga en BGR y la red se entrenó en ese orden. Solo se convierte a RGB para mostrar (Sesión 02: invertido, la imagen se ve casi bien y está mal). |
| **¿Qué mide SSIM que PSNR no?** | PSNR es solo error cuadrático. SSIM compara brillo, contraste y estructura en ventanas gaussianas 11×11; nuestro SSIM coincide con scikit-image. |

**Para cerrar:** *La CNN no reemplaza lo que aprendimos en el semestre: cada capa es el kernel de la Sesión 01, cada filtro es una neurona de la Sesión 11 y la pérdida se mide con lo que aprendimos en la Sesión 04.*

## Conexión con los talleres

| Sesión | Dónde está en el proyecto |
|---|---|
| 01 | `conv2d` propia, `np.clip` antes de uint8, transformación afín de brillo y contraste, kernel de realce |
| 02 | Orden BGR, pesos de luminancia, histogramas por canal en la app |
| 04 | Mediana y gaussiano como línea base; ruido generado con semilla para conservar la referencia |
| 05 | Magnitud del gradiente (Sobel) en la app para ver bordes y ruido |
| 11 | Cada filtro es una suma ponderada más sesgo; una sola capa lineal tiene un techo |
| 12 | Conteo de parámetros con (entradas + 1)·neuronas; apilar capas no lineales |

## Demo en vivo (3 minutos)

1. `.\.venv\Scripts\python.exe -m streamlit run app.py`
2. Modo experimento, foto de prueba, ruido gaussiano σ = 25: pestaña **Comparación** con PSNR y SSIM.
3. Pestaña **CNN**: mostrar "lo que la red estimó como ruido" y los 32 filtros aprendidos.
4. Cambiar a sal y pimienta: la mediana gana y la app lo advierte.
5. Cargar una foto propia en modo **foto real**: no aparecen PSNR ni SSIM, solo el sigma estimado.

## Limitaciones que debes decir antes de que te las pregunten

- El ruido real de cámara depende del brillo y no es exactamente gaussiano.
- BSDS500 son fotos pequeñas; fotos de celular de 12 MP tienen otra escala de ruido.
- La red solo quita ruido: brillo, contraste y nitidez son procesamiento tradicional y así se presentan.
