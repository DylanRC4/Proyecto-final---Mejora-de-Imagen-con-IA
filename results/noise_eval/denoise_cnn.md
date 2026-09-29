models\denoise_cnn.pt en el conjunto de prueba BSDS500: 200 fotos. Valores: PSNR medio / SSIM medio.

| Ruido | Sin filtro | Mediana | Gaussiano | CNN (nuestra) | CNN > gaussiano |
|---|---|---|---|---|---|
| gaussian_15 | 24.83 dB / 0.563 | 26.96 dB / 0.714 | 28.04 dB / 0.785 | 30.94 dB / 0.872 | 199/200 |
| gaussian_25 | 20.54 dB / 0.385 | 24.96 dB / 0.584 | 26.17 dB / 0.669 | 29.62 dB / 0.837 | 200/200 |
| gaussian_50 | 15.01 dB / 0.193 | 22.61 dB / 0.481 | 23.23 dB / 0.567 | 25.04 dB / 0.633 | 182/200 |
| salt_pepper_0.05 | 17.97 dB / 0.419 | 28.88 dB / 0.858 | 24.84 dB / 0.645 | 21.80 dB / 0.516 | 5/200 |
| camara_0.003 | 29.09 dB / 0.804 | 26.57 dB / 0.735 | 27.08 dB / 0.773 | 29.15 dB / 0.831 | 200/200 |
