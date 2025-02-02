import os
import zipfile
from pathlib import Path

# Rutas
data_path = Path("data/")

# Asegurar que la ruta es correcta usando Path()
dataset_path = Path(r"C:\Users\opera\OneDrive\Escritorio\Importante\Computer vision\ProyectoNaive\archive.zip")  # <- Se corrige la barra invertida

# If the image folder doesn't exist, download it and prepare it...
if data_path.is_dir():
    print(f"{data_path} directory exists.")
else:
    print(f"Did not find {data_path} directory, creating one...")
    data_path.mkdir(parents=True, exist_ok=True)


    # Unzip pizza, steak, sushi data
with zipfile.ZipFile(dataset_path, "r") as zip_ref:
    print("Unzipping data...")
    zip_ref.extractall(data_path)