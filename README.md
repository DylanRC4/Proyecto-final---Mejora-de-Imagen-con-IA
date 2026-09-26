# ImageEnhance AI

Software de mejoramiento de imágenes mediante procesamiento digital de imágenes y redes neuronales convolucionales.
Proyecto final de **Inteligencia Artificial II** — Institución Universitaria de Colombia.
Autor: Dylan Esteban Ricaurte Cuervo.

> Estado: en desarrollo.

## Objetivo

Eliminar ruido digital en fotografías con una CNN pequeña entrenada por nosotros, y compararla contra
filtros tradicionales (mediana y gaussiano) con métricas objetivas (PSNR y SSIM).

## Alcance y limitaciones

- El problema principal es el **ruido gaussiano aditivo**. El ruido real de una cámara depende de la
  señal y no es exactamente gaussiano, así que la mejora en fotos reales puede ser menor.
- PSNR y SSIM **solo se calculan cuando existe una imagen limpia de referencia** (ruido agregado de forma
  controlada). En fotos reales sin referencia no se reportan.
- Brillo, contraste y nitidez se ajustan con **procesamiento tradicional**, no con la CNN.
- El modelo se entrena desde cero. La arquitectura se inspira en DnCNN (Zhang et al., 2017);
  no se usan pesos preentrenados.

## Instalación (Windows, PowerShell)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pruebas

```powershell
python -m pytest -q
```
