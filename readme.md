# ⚽ Clasificación de Entradas en Fútbol: Falta vs Entrada Limpia

Proyecto final de la asignatura **Procesamiento Digital de Imágenes (PDI)**.

Resumen: entrenamos y desplegamos un modelo de **visión por computador** que clasifica imágenes de entradas/tackles en fútbol en dos clases:

- `Clean_Tackles` → entrada limpia
- `Fouls` → falta

El flujo general incluye entrenamiento en **PyTorch** (transfer learning con `ResNet18`), exportación a **TorchScript** y despliegue en un **HuggingFace Space** con UI en **Gradio**. Se incluyen scripts de inferencia local y vía API.

---

**Integrantes**

- Juan Esteban Agreda Gutiérrez
- Santiago Zambrano

---

## 🧠 Planteamiento del problema

En el contexto del fútbol moderno (VAR, revisiones de jugadas, etc.), es relevante analizar si una entrada es una falta o una recuperación de balón limpia.

Objetivo: dado un **frame de video o una fotografía** de una entrada, clasificar la jugada en una de dos clases:

1. **Clean_Tackles** – entrada limpia
2. **Fouls** – falta

Nota: el modelo es una herramienta de apoyo y un ejemplo aplicado de PDI y deep learning, no pretende reemplazar al árbitro.

---

## 📦 Dataset

Dataset usado: **Football Tackles** (Kaggle).

Organización (ejemplo):

```text
football-tackles/
├─ var_200/
│  └─ VAR/
│     ├─ Clean_Tackles/
│     └─ Fouls/
├─ var400/
├─ var500/
└─ var600/
```

En el notebook de entrenamiento se:

- Cargan todas las imágenes de `Clean_Tackles` y `Fouls`.
- Se dividen en `train` / `valid` / `test`.
- Se aplican transformaciones de aumento de datos y normalización.

---

## 🧰 Tecnologías utilizadas

- Python 3
- PyTorch + torchvision
- ResNet18 (transfer learning)
- TorchScript (exportación del modelo)
- Kaggle Notebooks (entrenamiento con GPU)
- Google Colab (pruebas y despliegue del Space)
- HuggingFace Spaces + Gradio (interfaz web)
- `gradio_client` (consumo de la API del Space)
- Matplotlib / Pillow (visualización)

---

## 📁 Estructura del proyecto

proyecto_PDI/
├─ notebooks/
│  ├─ clasificacion-futbol.ipynb        # Entrenamiento + evaluación + exportación TorchScript
│  └─ PDI_hf_space.ipynb                # Preparación y pruebas del Space en HuggingFace
│
├─ src/
│  ├─ local_inference.py                # Inferencia local con el modelo TorchScript
│  └─ api_inference.py                  # Inferencia vía API contra el Space de HuggingFace
│
├─ weights/
│  ├─ model_best.pth                    # Pesos del modelo entrenado en PyTorch
│  └─ model_futbol_ts.pt                # Modelo exportado a TorchScript (CPU)
│
├─ ejemplo/
│  └─ image.png                         # Imagen de ejemplo para pruebas
│
├─ requirements.txt                     # Dependencias para entorno local/Space
└─ README.md

---

## 🔄 Flujo del proyecto

1. Entrenamiento en Kaggle

- Notebook principal: `notebooks/clasificacion-futbol.ipynb`.
- Se usa GPU (T4) para entrenar una `ResNet18` finamente ajustada al dataset.
- Se guardan los pesos: `weights/model_best.pth` (mejor modelo en validación) y gráficas de pérdida/accuracy.

2. Exportación a TorchScript

- En el notebook se recrea la arquitectura para inferencia, se cargan `model_best.pth` y se genera el modelo TorchScript con `torch.jit.trace` → `weights/model_futbol_ts.pt`.
- Se compara precisión y tiempos de inferencia entre el modelo PyTorch y el modelo TorchScript.

3. Despliegue en HuggingFace Space

- Space: `juanAG1210/pdi-futbol-tackles` (código principal: `app.py`).
- El Space carga `model_futbol_ts.pt`, aplica las mismas transformaciones y muestra una interfaz en Gradio donde el usuario sube una imagen y el modelo devuelve la predicción y probabilidades por clase.

---

## 🔧 Scripts de inferencia

- `src/local_inference.py`:
	- Carga `weights/model_futbol_ts.pt` localmente.
	- Recibe una ruta de imagen.
	- Imprime la predicción y probabilidades.
	- Genera `resultado_local.png` con la imagen y la etiqueta superpuesta.

- `src/api_inference.py`:
	- Usa `gradio_client.Client` para conectarse al Space de HuggingFace.
	- Envía una imagen al endpoint `/predict`.
	- Intenta generar `resultado_api.png` con la predicción obtenida desde la API.

---

## 🚀 Cómo ejecutar inferencias localmente

1. Crear entorno virtual e instalar dependencias

Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate         # Linux/WSL
# En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

2. Inferencia local con TorchScript

Asegúrate de que `weights/model_futbol_ts.pt` exista y que tengas una imagen en `ejemplo/image.png` (o usa imágenes del dataset).

Ejecuta:

```bash
python src/local_inference.py --image ejemplo/image.png
```

Salida esperada (ejemplo):

```
[INFO] Usando dispositivo: cpu
[INFO] Cargando modelo desde: weights/model_futbol_ts.pt
[INFO] Abriendo imagen: ejemplo/image.png
[RESULTADO] Predicción: Clean_Tackles
[RESULTADO] Probabilidades: {'Clean_Tackles': 0.62, 'Fouls': 0.38}
[INFO] Gráfico guardado en: resultado_local.png
```

Se genera `resultado_local.png` con la imagen y la predicción.

3. Inferencia vía API (HuggingFace Space)

En `src/api_inference.py` se define el ID del Space:

```py
SPACE_ID = "juanAG1210/pdi-futbol-tackles"
```

Ejecuta:

```bash
python src/api_inference.py --image ejemplo/image.png
```

El script conecta con el Space, envía la imagen y trata de dibujar la predicción en `resultado_api.png`.
Nota: si el Space tiene un error, el script lo reportará; la lógica del cliente está lista para consumir la API cuando el servidor responde correctamente.

---

## 📊 Resultados (resumen)

En el notebook de entrenamiento se registran, para el conjunto de validación/test:

- Accuracy
- Pérdida (loss)
- Matriz de confusión
- Curvas de evolución de pérdida y accuracy por época

De forma cualitativa, el modelo logra diferenciar razonablemente bien entre entradas limpias y faltas con base en imágenes estáticas. El desempeño se ve consistente entre entrenamiento y validación (sin sobreajuste extremo).

(Los valores numéricos exactos pueden verse en las celdas finales del notebook `clasificacion-futbol.ipynb`.)

---

## 🔍 Limitaciones y trabajo futuro

- El modelo opera sobre frames individuales y no considera secuencias de video → no ve el contexto completo de la jugada.
- El dataset es relativamente limitado y puede no cubrir todos los tipos de entradas.
- No se consideran señales adicionales (posición del balón, velocidad, etc.).

Posibles extensiones:

- Usar modelos 3D o basados en video (CNN + LSTM, Transformers, etc.).
- Entrenar con más clases (mano, juego peligroso, etc.).
- Integración con sistemas de análisis táctico o herramientas para árbitros.

---

## 📚 Cómo relaciona con la asignatura PDI

El proyecto integra varios conceptos vistos en clase:

- Preprocesamiento de imágenes y normalización.
- Uso de arquitecturas convolucionales (CNNs).
- Entrenamiento y evaluación de modelos de clasificación.
- Exportación y despliegue de modelos (TorchScript, HuggingFace, APIs).
- Construcción de pipelines de PDI “del laboratorio al despliegue”.

---

## 📝 Referencias

- Documentación de PyTorch: https://pytorch.org/
- Documentación de torchvision: https://pytorch.org/vision/stable/index.html
- Gradio + HuggingFace Spaces: https://www.gradio.app/ y https://huggingface.co/spaces
- Dataset Football Tackles en Kaggle.