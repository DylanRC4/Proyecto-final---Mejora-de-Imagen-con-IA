models\detail_cnn_v2.pt en 200 fotos de prueba (bsds500). PSNR medio / SSIM medio. Real-ESRGAN es externo y generativo.

| Condición | sin_restaurar | nitidez_clasica | cnn_detalle | real_esrgan | CNN mejora la foto |
|---|---|---|---|---|---|
| fija | 25.82 dB / 0.729 | 26.16 dB / 0.735 | 26.51 dB / 0.760 | 26.24 dB / 0.749 | 199/200 |
| desenfoque | 25.79 dB / 0.742 | 26.99 dB / 0.797 | 30.59 dB / 0.907 | 26.33 dB / 0.750 | 200/200 |
| jpeg | 28.21 dB / 0.827 | 27.48 dB / 0.814 | 28.34 dB / 0.833 | 26.45 dB / 0.789 | 163/200 |
| baja_res | 28.42 dB / 0.853 | 29.03 dB / 0.881 | 28.97 dB / 0.876 | 27.25 dB / 0.810 | 185/200 |
| fuerte | 22.94 dB / 0.576 | 23.09 dB / 0.575 | 23.48 dB / 0.599 | 23.90 dB / 0.617 | 200/200 |
| celular | 28.80 dB / 0.796 | 27.61 dB / 0.765 | 28.64 dB / 0.787 | 25.95 dB / 0.767 | 60/200 |
