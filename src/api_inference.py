import argparse
import os

from gradio_client import Client
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # backend sin ventana gráfica (ideal en WSL)
import matplotlib.pyplot as plt

# ⬇⬇⬇ PON AQUÍ EL ID REAL DE TU SPACE ⬇⬇⬇
# Ejemplo: si tu Space es https://huggingface.co/spaces/juanAG1210/pdi-futbol-tackles
# entonces:
SPACE_ID = "juanAG1210/pdi-futbol-tackles"


from gradio_client import Client

SPACE_ID = "juanAG1210/pdi-futbol-tackles"


def call_space(image_path: str):
    """Llama al Space de HuggingFace con la imagen y devuelve la respuesta."""
    print(f"[INFO] Creando cliente para el Space: {SPACE_ID}")
    client = Client(SPACE_ID)

    print(f"[INFO] Endpoints disponibles en el Space (view_api):")
    try:
        client.view_api()  # solo imprime info útil en consola
    except Exception as e:
        print(f"[WARN] No se pudo listar la API: {e}")

    print(f"[INFO] Enviando imagen al Space: {image_path}")
    try:
        # 🔹 La mayoría de Interfaces simples usan api_name="/predict"
        try:
            result = client.predict(image_path, api_name="/predict")
        except Exception as e1:
            print(f"[WARN] /predict falló ({e1}), probando endpoint por defecto...")
            # 🔹 Fallback: sin api_name (usa el único endpoint disponible)
            result = client.predict(image_path)
    except Exception as e:
        print(f"[ERROR] Falló la llamada al Space: {e}")
        return None

    print(f"[DEBUG] Respuesta cruda del Space: {result}")
    return result




def procesar_respuesta(result, image_path: str):
    """Interpreta la respuesta del Space y guarda una imagen con el resultado."""
    if result is None:
        print("[ERROR] No hay resultado para procesar.")
        return

    # En nuestro Space la interfaz devuelve [pred_label, prob_dict]
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        pred_label = str(result[0])
        prob_dict = result[1]
    else:
        pred_label = str(result)
        prob_dict = {}

    print(f"[RESULTADO API] Predicción: {pred_label}")
    print(f"[RESULTADO API] Probabilidades: {prob_dict}")

    # Cargar la imagen original
    image = Image.open(image_path).convert("RGB")

    # Graficar y guardar
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(f"API: {pred_label}")
    plt.tight_layout()

    out_path = "resultado_api.png"
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[INFO] Gráfico de la API guardado en: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen de entrada (ejemplo: ejemplo/image.png)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] No se encontró la imagen en {args.image}")
        return

    print(f"[INFO] Usando el Space: {SPACE_ID}")
    result = call_space(args.image)
    procesar_respuesta(result, args.image)


if __name__ == "__main__":
    main()
