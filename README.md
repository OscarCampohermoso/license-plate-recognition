# License Plate Recognition Project

Este proyecto utiliza un modelo de reconocimiento de matrículas de vehículos. Asegúrate de seguir los pasos a continuación para configurar y ejecutar el proyecto correctamente.

## Requisitos

- Python 3.8 o superior
- Las siguientes bibliotecas de Python:

  ```bash
  pip install opencv-python glob2 torch ultralytics gdown
  ```

> **Nota:** Se recomienda utilizar un entorno virtual, pero no es esencial.

## Descarga del Dataset

Primero, descarga el dataset de reconocimiento de matrículas desde el siguiente enlace:

[Descargar Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)

Asegúrate de que los archivos descargados se encuentren en la raíz del proyecto.

## Uso

Después de haber instalado las dependencias y descargado el dataset, puedes empezar a utilizar el proyecto. Asegúrate de tener todos los archivos necesarios en la estructura correcta.


### Probar la api
dependencias para instalar:

```bash
pip install fastapi uvicorn python-multipart pillow easyocr ultralytics opencv-python
```

* Levantar el servidor:
```bash
uvicorn main:app --reload
```

* para probar la api se debe hacer correr el archivo api_test.py, se generará "output.jpg" como salida

si no se dibujaran los rectangulos:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
pip install opencv-python --upgrade

```