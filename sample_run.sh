
# Example: create virtual env, install requirements, and run training
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
python train.py --data_dir data/genres --epochs 30 --batch_size 16
