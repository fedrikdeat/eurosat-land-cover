"""Saqlanǵan EuroSAT modeli menen suwretti klassifikaciyalaw.

Paydalanıw:  python predict.py photo.jpg [model.keras] [class_names.json]
"""
import json
import sys

import numpy as np
from PIL import Image
from tensorflow import keras

IMG_SIZE = (128, 128)


def main():
    if len(sys.argv) < 2:
        sys.exit("Paydalanıw: python predict.py photo.jpg [model.keras] [class_names.json]")
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "eurosat_model.keras"
    names_path = sys.argv[3] if len(sys.argv) > 3 else "class_names.json"

    model = keras.models.load_model(model_path)
    with open(names_path, encoding="utf-8") as f:
        class_names = json.load(f)

    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    probs = model.predict(np.array(img, dtype="float32")[None], verbose=0)[0]

    print(f"Suwret: {image_path}")
    for i in probs.argsort()[::-1][:5]:
        print(f"  {class_names[i]:22s} {probs[i] * 100:5.1f}%")


if __name__ == "__main__":
    main()
