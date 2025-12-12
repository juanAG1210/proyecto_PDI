import argparse
import os

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica (ideal para WSL)
import matplotlib.pyplot as plt

# Configuración básica
IMG_SIZE = 224
CLASS_NAMES = ["Clean_Tackles", "Fouls"]
DEVICE = torch.device("cpu")
MODEL_PATH = os.path.join("weights", "model_futbol_ts.pt")


def load_model():
    """Carga el modelo TorchScript desde disco."""
    print(f"[INFO] Cargando modelo desde: {MODEL_PATH}")
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model


# Transformaciones igual que en entrenamiento
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(model, image_path: str):
    """Hace la predicción sobre una imagen y guarda un gráfico con el resultado."""
    print(f"[INFO] Abriendo imagen: {image_path}")
    image = Image.open(image_path).convert("RGB")

    x = transform(image).unsqueeze(0)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(x.to(DEVICE))
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = CLASS_NAMES[pred_idx]
    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    print(f"[RESULTADO] Predicción: {pred_label}")
    print(f"[RESULTADO] Probabilidades: {prob_dict}")

    # ----- Guardar figura en PNG en vez de mostrarla -----
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")
    title = (
        f"{pred_label} | "
        f"Clean_Tackles: {prob_dict['Clean_Tackles']:.2f} | "
        f"Fouls: {prob_dict['Fouls']:.2f}"
    )
    ax.set_title(title)

    out_path = "resultado_local.png"
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    print(f"[INFO] Gráfico guardado en: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen de entrada (por ejemplo: ejemplos/image.jpg)",
    )
    args = parser.parse_args()

    # Comprobaciones básicas
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] No se encontró el modelo en {MODEL_PATH}")
        return

    if not os.path.exists(args.image):
        print(f"[ERROR] No se encontró la imagen en {args.image}")
        return

    print(f"[INFO] Usando dispositivo: {DEVICE}")

    model = load_model()
    predict_image(model, args.image)


if __name__ == "__main__":
    main()
