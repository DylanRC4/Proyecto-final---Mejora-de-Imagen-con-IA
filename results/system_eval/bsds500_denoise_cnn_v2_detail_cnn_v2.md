Sistema automático en 200 fotos de prueba (bsds500), CNN de ruido models/denoise_cnn_v2.pt y de detalle models\detail_cnn_v2.pt. PSNR medio (limitado a 60 dB) / SSIM medio. Mejora o empeora: cambio de PSNR mayor de 0,05 dB.

| Condición | Sin procesar | Sistema (nuestras CNN) | Sistema clásico | Mejora / empeora | Plan más común |
|---|---|---|---|---|---|
| limpia | 60.00 dB / 1.000 | 53.52 dB / 0.986 | 52.53 dB / 0.973 | 0/200 / 45/200 | (nada) |
| ruido | 24.83 dB / 0.563 | 31.00 dB / 0.869 | 27.37 dB / 0.769 | 191/200 / 7/200 | ruido |
| oscura_ruidosa | 12.97 dB / 0.473 | 21.94 dB / 0.719 | 21.31 dB / 0.643 | 191/200 / 9/200 | ruido → contraste → luz |
| desenfoque_jpeg | 26.10 dB / 0.745 | 25.79 dB / 0.755 | 25.34 dB / 0.740 | 173/200 / 25/200 | detalle |
| oscura_velo | 14.30 dB / 0.787 | 23.99 dB / 0.889 | 23.94 dB / 0.887 | 183/200 / 15/200 | contraste → luz |
| dominante | 24.46 dB / 0.978 | 31.65 dB / 0.980 | 31.22 dB / 0.969 | 186/200 / 8/200 | color |
| todo | 14.63 dB / 0.569 | 18.06 dB / 0.611 | 18.21 dB / 0.608 | 144/200 / 15/200 | ruido → detalle → contraste → luz |
| celular | 28.80 dB / 0.796 | 29.64 dB / 0.849 | 27.09 dB / 0.779 | 157/200 / 9/200 | ruido |
| fuerte | 22.94 dB / 0.576 | 21.87 dB / 0.582 | 21.13 dB / 0.559 | 147/200 / 50/200 | detalle |
