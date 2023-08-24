# Thesis
## Install the necessary libraries :
conda create --name myenv python=3.7
conda activate myenv
pip install -r requirements.txt

## Then install the following libraries, not registered on Pypi :
https://github.com/mxfold/mxfold2 (check releases at https://github.com/mxfold/mxfold2/releases)
https://github.com/lipan6461188/IPyRSSA

## Then, you can run the script to launch predictions :
python main_run_preds.py