# wildfire-prediction-pytorch
Individual project assignment, Fördjupning i Pythonprogrammering (YH Data Science)

## Environment
- Python 3.13.13
- Packages: Pandas, Numpy, Matplotlib, Scikit-learn, PyTorch, Torchvision, Streamlit, Pillow, kagglehub (see requirements.txt)

## Data
- Wildfire Prediction Dataset (Kaggle), downloaded automatically via kagglehub — not included in the repo
- Requires a Kaggle API key: go to kaggle.com → Account → Create New API Token, which downloads a `kaggle.json` file.
Place it at `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\kaggle.json` on Windows). This file is not included in the repository.

## Getting started
```
# clone the project
git clone https://github.com/MalinMo/wildfire-prediction-pytorch.git
 
# create and activate a virtual environment
python -m venv .venv
 
# install dependencies
python -m pip install -r requirements.txt
```

## Run
### Streamlit app (demo)
```
streamlit run app.py
```

### Training and evaluation
Open wildfire_prediction.ipynb in Jupyter/JupyterLab

## Rapport och presentation
- Rapport_Uppgift2_Malin_Moisander.docx
- Presentation_Uppgift2_Malin_Moisander.pptx