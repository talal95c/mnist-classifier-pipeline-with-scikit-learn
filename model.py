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

# Step 2 - binary_target (not yet solved)
# TODO: implement

# Step 3 - train_sgd (not yet solved)
# TODO: implement

# Step 4 - cross_val_predictions (not yet solved)
# TODO: implement

# Step 5 - confusion_counts (not yet solved)
# TODO: implement

# Step 6 - precision_recall_f1 (not yet solved)
# TODO: implement

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

