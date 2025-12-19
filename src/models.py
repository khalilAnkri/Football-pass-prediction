import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

def train_model(X_features, y_target):
    print("Training Gradient Boosting with Calibration...")
    
    # Base Model: Gradient Boosting
    # Optimized params for typical football data
    gb = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    
    # Calibration Wrapper
    # 'isotonic' is good for large data, 'sigmoid' for smaller. 
    clf = CalibratedClassifierCV(gb, method='isotonic', cv=3)
    
    clf.fit(X_features, y_target)
    return clf

def predict_pass_probabilities(model, X_features, original_indices):
    # (Keep this function exactly the same as before)
    raw_probas = model.predict_proba(X_features)[:, 1]
    results = pd.DataFrame({'pass_id': original_indices, 'raw_proba': raw_probas})
    
    def normalize_group(group):
        total = group['raw_proba'].sum()
        if total == 0: return group
        group['normalized_proba'] = group['raw_proba'] / total
        return group

    results = results.groupby('pass_id', group_keys=False).apply(normalize_group)
    return results

def evaluate_predictions(y_true, probas_df):
    # The Brier Score calculation here is correct for the competition
    pass_ids = y_true.index.unique()
    y_true_sorted = y_true.loc[pass_ids]
    
    predicted_receivers = []
    prob_arrays = []
    
    for pid in pass_ids:
        pass_data = probas_df[probas_df['pass_id'] == pid].reset_index(drop=True)
        
        p_slice = pass_data['normalized_proba'].values
        
        best_candidate_idx = np.argmax(p_slice) 
        predicted_receiver = best_candidate_idx + 1
        
        predicted_receivers.append(predicted_receiver)
        prob_arrays.append(p_slice)
        
    acc = accuracy_score(y_true_sorted, predicted_receivers)
    
    n_samples = len(pass_ids)
    y_true_onehot = np.zeros((n_samples, 22))
    for i, receiver_id in enumerate(y_true_sorted.values.flatten()):
         # Safety check for receiver_id being 1-22
         idx = int(receiver_id) - 1
         if 0 <= idx < 22:
             y_true_onehot[i, idx] = 1
         
    bs = np.mean(np.sum((y_true_onehot - np.array(prob_arrays))**2, axis=1))
    
    return acc, bs