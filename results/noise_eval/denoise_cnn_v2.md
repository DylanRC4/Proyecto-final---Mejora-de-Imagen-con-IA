models\denoise_cnn_v2.pt en el conjunto de prueba BSDS500: 200 fotos. Valores: PSNR medio / SSIM medio.

| Ruido | Sin filtro | Mediana | Gaussiano | CNN (nuestra) | CNN > gaussiano |
|---|---|---|---|---|---|
| gaussian_15 | 24.83 dB / 0.563 | 26.96 dB / 0.714 | 28.04 dB / 0.785 | 31.68 dB / 0.877 | 200/200 |
| gaussian_25 | 20.54 dB / 0.385 | 24.96 dB / 0.584 | 26.17 dB / 0.669 | 29.11 dB / 0.812 | 200/200 |
| gaussian_50 | 15.01 dB / 0.193 | 22.61 dB / 0.481 | 23.23 dB / 0.567 | 25.20 dB / 0.644 | 194/200 |
| salt_pepper_0.05 | 17.97 dB / 0.419 | 28.88 dB / 0.858 | 24.84 dB / 0.645 | 19.51 dB / 0.464 | 0/200 |
| camara_0.003 | 29.09 dB / 0.804 | 26.57 dB / 0.735 | 27.08 dB / 0.773 | 30.97 dB / 0.879 | 200/200 |
