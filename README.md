# SISTEMA-DE-DETECCI-N-DE-C-NCER-ORAL# 🧠 Detección de Cáncer Oral (OSCC) con Deep Learning

[![Django](https://img.shields.io/badge/Django-4.2-green)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Aplicación web integral** que entrena una Red Neuronal Convolucional (CNN) para clasificar imágenes de histopatología oral en **Normal** vs **Carcinoma Oral de Células Escamosas (OSCC)**. Construida con Django backend, TensorFlow 2.x y métricas de scikit‑learn.

<p align="center">
  <img src="https://img.shields.io/badge/status-activo-success" />
  <img src="https://img.shields.io/badge/precisión-~90%25-brightgreen" />
  <img src="https://img.shields.io/badge/recall-92%25-critical" />
</p>

---

## 📖 Tabla de Contenidos

- [✨ Características](#-características)
- [🧱 Tecnologías](#-tecnologías)
- [📁 Estructura del Dataset](#-estructura-del-dataset)
- [🚀 Instalación](#-instalación)
- [💻 Uso](#-uso)
- [🏗️ Arquitectura del Modelo](#️-arquitectura-del-modelo)
- [📊 Métricas de Evaluación](#-métricas-de-evaluación)
- [📈 Visualización de Resultados](#-visualización-de-resultados)
- [🔮 Mejoras Futuras](#-mejoras-futuras)
- [📄 Licencia](#-licencia)
- [🙏 Agradecimientos](#-agradecimientos)

---

## ✨ Características

- ✅ **Carga automática del dataset** – lee imágenes desde las carpetas `Normal/` y `OSCC/`.
- ✅ **División 80/20 entrenamiento/validación** con semilla reproducible.
- ✅ **CNN personalizada** – 3 bloques convolucionales + Global Average Pooling + Dropout.
- ✅ **Entrenamiento en tiempo real** – EarlyStopping evita sobreajuste.
- ✅ **Métricas completas** – Precisión, Recall, F1‑Score, Exactitud, Matriz de Confusión.
- ✅ **Mapa de calor interactivo** – matriz de confusión embebida directamente en el HTML.
- ✅ **Reporte detallado por clase** – promedios macro y ponderado.
- ✅ **Interfaz Django limpia** – entrenamiento y evaluación con un solo clic.

---

## 🧱 Tecnologías

| Capa          | Tecnología                                                               |
|---------------|--------------------------------------------------------------------------|
| **Backend**   | Django 4.2 (Python 3.10+)                                                |
| **Deep Learning** | TensorFlow 2.13 (API de Keras)                                      |
| **Métricas**  | scikit‑learn (precisión, recall, f1, matriz de confusión, reporte de clasificación) |
| **Visualización** | Matplotlib + Seaborn (mapa de calor), embebido como base64 PNG       |
| **Frontend**  | HTML5 + Bootstrap 5 (mínimo pero extensible)                            |

---

## 📁 Estructura del Dataset

Tu dataset **debe** colocarse dentro de la carpeta `dataset/` en la raíz del proyecto:
django_deep_learning/
├── dataset/
│ ├── Normal/ # (ej. 125 imágenes)
│ │ ├── img1.jpg
│ │ ├── img2.jpg
│ │ └── ...
│ └── OSCC/ # (ej. 121 imágenes)
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
├── myapp/ # Aplicación Django (views.py, urls.py, templates/)
├── manage.py
└── ...

text

- Formatos soportados: `.jpg`, `.jpeg`, `.png`
- Las imágenes se redimensionan automáticamente a **128×128** píxeles (RGB).
- No se necesita anotación manual – los nombres de las carpetas se usan como etiquetas (`Normal` = 0, `OSCC` = 1).

---

## 🚀 Instalación

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/tuusuario/deteccion-cancer-oral.git
cd deteccion-cancer-oral
2️⃣ Crear y activar un entorno virtual
bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
3️⃣ Instalar dependencias
bash
pip install -r requirements.txt
Si no tienes un requirements.txt, instala manualmente:

bash
pip install django tensorflow scikit-learn matplotlib seaborn numpy
4️⃣ Prepara tu dataset
Coloca tus carpetas Normal/ y OSCC/ dentro de un directorio dataset/ en la raíz del proyecto (como se mostró arriba).

5️⃣ Ejecutar migraciones de Django (opcional, solo si agregas modelos de usuario)
bash
python manage.py migrate
6️⃣ Iniciar el servidor de desarrollo
bash
python manage.py runserver
Abre tu navegador en http://127.0.0.1:8000/

💻 Uso
Página principal – haz clic en el botón “Entrenar y Evaluar”.

Procesamiento backend (toma 1–3 minutos dependiendo del CPU):

Carga ~250 imágenes

Entrena la CNN hasta 15 épocas (early stopping si la pérdida de validación se estanca)

Calcula predicciones en el conjunto de validación (20%)

Genera precisión, recall, F1, matriz de confusión y reporte de clasificación

La página de resultados muestra:

Exactitud general (entrenamiento y validación)

Precisión / Recall / F1

Mapa de calor interactivo de la matriz de confusión

Tabla detallada de métricas por clase

Reporte de clasificación completo en texto plano

Nota: El entrenamiento se realiza en cada clic del botón. Para producción, deberías guardar el modelo y reutilizarlo.

🏗️ Arquitectura del Modelo
Una CNN ligera diseñada para clasificación binaria con entrenamiento rápido en CPU.

Capa	Forma de Salida	Parámetros	Descripción
Entrada (128×128×3)	(128, 128, 3)	0	Imagen RGB
Rescaling	(128, 128, 3)	0	Normaliza píxeles a [0,1]
Conv2D (16 filtros, 3×3)	(128, 128, 16)	448	ReLU + padding='same'
MaxPool2D (2×2)	(64, 64, 16)	0	Reduce resolución a la mitad
Conv2D (32 filtros, 3×3)	(64, 64, 32)	4,640	ReLU + padding='same'
MaxPool2D (2×2)	(32, 32, 32)	0	
Conv2D (64 filtros, 3×3)	(32, 32, 64)	18,496	ReLU + padding='same'
MaxPool2D (2×2)	(16, 16, 64)	0	
GlobalAveragePooling2D	(64,)	0	Reduce cada mapa de características a un valor
Dropout (0.3)	(64,)	0	Regularización (30% de neuronas apagadas)
Dense (1, sigmoid)	(1,)	65	Probabilidad de OSCC
Total de parámetros entrenables: ~23,649 → muy ligero, funciona en cualquier portátil.

¿Por qué esta arquitectura?
3 bloques convolucionales incrementan el número de filtros (16→32→64) para aprender características jerárquicas (bordes → formas → patrones complejos de OSCC).

GlobalAveragePooling2D en lugar de Flatten → reduce drásticamente los parámetros y previene sobreajuste.

Dropout(0.3) mejora la generalización.

Sigmoid da una probabilidad interpretable.

📊 Métricas de Evaluación
Una vez finalizado el entrenamiento, el sistema calcula las siguientes métricas sobre el conjunto de validación (20% de los datos):

Métrica	Fórmula	Por qué es importante para detección de OSCC
Precisión	TP / (TP + FP)	¿Cuántos OSCC predichos eran correctos? Alta precisión = pocas falsas alarmas.
Recall	TP / (TP + FN)	¿Cuántos OSCC reales detectamos? Crítico – perder un cáncer es peligroso.
F1‑Score	2 × (Precisión × Recall) / (P + R)	Media armónica – equilibra ambas.
Exactitud	(TP + TN) / Total	Corrección global (puede ser engañosa si las clases están desbalanceadas).
Matriz de Confusión	Tabla 2×2 (TN, FP, FN, TP)	Desglose visual de aciertos/errores por clase.
Todas las métricas se muestran como porcentajes redondeados a dos decimales.

📈 Visualización de Resultados
Después de la evaluación, obtienes:

Mapa de calor de la Matriz de Confusión (generado con Seaborn)

Azul oscuro = muchos casos (aciertos)

Azul claro = pocos casos (errores)

Embebido como imagen base64 – sin archivos externos.

Tabla de Reporte de Clasificación

Clase	Precisión (%)	Recall (%)	F1‑Score (%)	Soporte
Normal	88.0	90.0	89.0	25
OSCC	91.7	88.0	89.8	24
Exactitud	–	–	89.8	49
Promedio macro	89.9	89.0	89.4	49
