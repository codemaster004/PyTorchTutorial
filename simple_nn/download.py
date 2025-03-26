import kagglehub
import shutil
import os

os.makedirs("./data", exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("shantanugarg274/heart-prediction-dataset-quantum", force_download=True)
shutil.copytree(path, "./", dirs_exist_ok=True)
shutil.move("Heart Prediction Quantum Dataset.csv", "heart-prediction-dataset.csv")
