Sistema automático en 200 fotos de prueba (bsds500), CNN de detalle models/detail_cnn_v2.pt. PSNR medio (limitado a 60 dB) / SSIM medio. Mejora o empeora: cambio de PSNR mayor de 0,05 dB.

| Condición | Sin procesar | Sistema (nuestras CNN) | Sistema clásico | Mejora / empeora | Plan más común |
|---|---|---|---|---|---|
| limpia | 60.00 dB / 1.000 | 55.23 dB / 0.985 | 55.18 dB / 0.984 | 0/200 / 28/200 | (nada) |
| ruido | 24.83 dB / 0.563 | 30.42 dB / 0.867 | 27.41 dB / 0.774 | 195/200 / 5/200 | ruido |
| oscura_ruidosa | 12.97 dB / 0.473 | 21.39 dB / 0.731 | 21.29 dB / 0.652 | 192/200 / 8/200 | ruido → contraste → luz |
| desenfoque_jpeg | 26.10 dB / 0.745 | 25.83 dB / 0.762 | 25.40 dB / 0.744 | 177/200 / 22/200 | detalle |
| oscura_velo | 14.30 dB / 0.787 | 24.00 dB / 0.890 | 24.00 dB / 0.890 | 182/200 / 16/200 | contraste → luz |
| dominante | 24.46 dB / 0.978 | 31.85 dB / 0.982 | 31.89 dB / 0.983 | 186/200 / 8/200 | color |
| todo | 14.63 dB / 0.569 | 18.39 dB / 0.592 | 18.32 dB / 0.530 | 130/200 / 14/200 | detalle → contraste → luz |
