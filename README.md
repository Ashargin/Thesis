# DivideFold

## Install the necessary libraries :
``` console
conda create --name myenv python=3.7
conda activate myenv
pip install -r requirements.txt
```

## Then install the following libraries, not registered on Pypi :

* [MXFold2](https://github.com/mxfold/mxfold2)


## Then, to use our tool :

### You can predict a sequence's secondary structure using the prediction function :
``` python
from DivideFold.predict import dividefold_predict
pred, _, _, _, _ = dividefold_predict(sequence)
```

### The default prediction tool to be applied after partitioning is MXfold2. However, our tool can use any function you like for the structure prediction part. If you would like to use a custom structure prediction function, you can use :
``` python
from DivideFold.predict import dividefold_predict
pred, _, _, _, _ = dividefold_predict(sequence, predict_fnc=my_structure_prediction_function)
```

### If you're only interested in the cut points, you can use :
``` python
from DivideFold.predict import dividefold_predict
cuts, _, _, _, _ = dividefold_predict(sequence, return_cuts=True)
```
This will return the cut points at the final step in the recursive cutting process.

### Or if you only want the cut points for the next step :
``` python
from DivideFold.predict import dividefold_get_cuts
cuts, _ = dividefold_get_cuts(sequence)
```
