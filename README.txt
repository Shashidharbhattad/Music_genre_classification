
Music Genre Classification System - Project
-----------------------------------------

Contents:
- report.txt : Project report
- notebook.ipynb : Colab/Jupyter-ready notebook with preprocessing, model, training, and evaluation cells
- utils.py : helper functions for loading audio & extracting features
- model.py : Keras model definitions (CNN & CRNN)
- train.py : training script (uses local dataset folder structure)
- requirements.txt : Python packages needed
- sample_run.sh : example commands to run locally

Notes:
- This project expects the GTZAN dataset organized as:
  data/genres/<genre_name>/*.wav
  (Each genre folder contains audio files.)
- If you use Google Colab you can upload the dataset or mount Google Drive.
- See notebook.ipynb for step-by-step runnable cells.
