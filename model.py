"""
MNIST Classifier Pipeline with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_mnist
import os
import tempfile
import urllib.request
import numpy as np

def load_mnist(n_train=10000, n_test=2000):
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")
    
    # 1. Télécharger si le fichier n'existe pas encore
    if not os.path.exists(path):
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        urllib.request.urlretrieve(url, path)
        
    # 2. with np.load() permet de fermer proprement le fichier après la lecture
    with np.load(path) as d:
        # 3. Les clés d'origine sont en minuscules
        X_train = d["x_train"][:n_train].reshape(n_train, -1).astype(np.float32)
        # Pas de reshape pour les labels (ils restent en 1D)
        y_train = d["y_train"][:n_train].astype(np.int64)
        
        # 4. On utilise n_test pour la partie test
        X_test = d["x_test"][:n_test].reshape(n_test, -1).astype(np.float32)
        y_test = d["y_test"][:n_test].astype(np.int64)
        
    # 5. On retourne bien un dictionnaire avec les bonnes clés
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }

# Step 2 - binary_target
def binary_target(y, digit=5):
    # TODO: boolean array, True where y == digit.

    return np.asarray(y)==digit

# Step 3 - train_sgd
from sklearn.linear_model import SGDClassifier
def train_sgd(X, y, random_state=42):
    # TODO: fit and return SGDClassifier(random_state=random_state).
    
    return SGDClassifier(random_state=random_state).fit(X,y)

# Step 4 - cross_val_predictions
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone 
def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    # TODO: cross_val_predict on a clone of clf with the given method.
    return cross_val_predict(clone(clf),X,y,cv=cv,method = method )

# Step 5 - confusion_counts
from sklearn.metrics import confusion_matrix 

def confusion_counts(y_true, y_pred):
    # TODO: {'TN': ..., 'FP': ..., 'FN': ..., 'TP': ...} from confusion_matrix(labels=[False, True]).
    cm =  confusion_matrix(y_true,y_pred,labels=[False,True])
    TN,FP,FN,TP  = cm.ravel()
    return {'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP)}

# Step 6 - precision_recall_f1
from sklearn.metrics import f1_score,recall_score,precision_score

def precision_recall_f1(y_true, y_pred):
    # TODO: (precision, recall, f1) via sklearn.metrics with zero_division=0.
    f1 = f1_score(y_true,y_pred,zero_division=0)
    r = recall_score(y_true,y_pred,zero_division=0)
    p = precision_score(y_true,y_pred,zero_division=0)

    return (float(p), float(r), float(f1))

# Step 7 - threshold_for_precision (not yet solved)
# TODO: implement

# Step 8 - evaluate_at_threshold (not yet solved)
# TODO: implement

# Step 9 - roc_auc (not yet solved)
# TODO: implement

# Step 10 - multiclass_pipeline (not yet solved)
# TODO: implement

# Step 11 - multiclass_cv_accuracy (not yet solved)
# TODO: implement

# Step 12 - normalized_confusion (not yet solved)
# TODO: implement

# Step 13 - most_confused_pairs (not yet solved)
# TODO: implement

# Step 14 - multilabel_targets (not yet solved)
# TODO: implement

# Step 15 - multilabel_knn (not yet solved)
# TODO: implement

# Step 16 - final_test_accuracy (not yet solved)
# TODO: implement

# Step 17 - save_and_reload_classifier (not yet solved)
# TODO: implement

# Step 18 - predict_digits (not yet solved)
# TODO: implement

