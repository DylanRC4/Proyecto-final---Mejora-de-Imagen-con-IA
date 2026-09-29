Sistema automático en 200 fotos de prueba (bsds500), CNN de ruido models/denoise_cnn_v2.pt y de detalle models/detail_cnn_v3.pt. PSNR medio (limitado a 60 dB) / SSIM medio. Mejora o empeora: cambio de PSNR mayor de 0,05 dB.

| Condición | Sin procesar | Sistema (nuestras CNN) | Sistema clásico | Mejora / empeora | Plan más común |
|---|---|---|---|---|---|
| limpia | 60.00 dB / 1.000 | 55.46 dB / 0.986 | 54.44 dB / 0.973 | 0/200 / 33/200 | (nada) |
| ruido | 24.83 dB / 0.563 | 31.16 dB / 0.868 | 27.56 dB / 0.769 | 194/200 / 5/200 | ruido |
| oscura_ruidosa | 12.97 dB / 0.473 | 21.93 dB / 0.714 | 21.31 dB / 0.643 | 190/200 / 10/200 | ruido → contraste → luz |
| desenfoque_jpeg | 26.10 dB / 0.745 | 25.82 dB / 0.746 | 25.41 dB / 0.740 | 167/200 / 29/200 | detalle |
| oscura_velo | 14.30 dB / 0.787 | 24.01 dB / 0.888 | 23.96 dB / 0.888 | 183/200 / 15/200 | contraste → luz |
| dominante | 24.46 dB / 0.978 | 24.04 dB / 0.965 | 23.90 dB / 0.954 | 0/200 / 28/200 | (nada) |
| todo | 14.63 dB / 0.569 | 18.05 dB / 0.600 | 18.21 dB / 0.608 | 157/200 / 16/200 | ruido → detalle → contraste → luz |
| celular | 28.80 dB / 0.796 | 29.70 dB / 0.849 | 27.13 dB / 0.779 | 159/200 / 6/200 | ruido |
| fuerte | 22.94 dB / 0.576 | 22.78 dB / 0.614 | 21.59 dB / 0.560 | 168/200 / 32/200 | detalle |
