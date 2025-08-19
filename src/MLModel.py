from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

class Model():
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        if model_type =='random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
            # decision tree
        else: 
            self.model = DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42
            )
    # note do not need a train function as you are retraining everytime throughout LOSO (if you're using the same exact dataset, use train function to reduce redundant training calls)
    def loso(self, all_data):
        patient_results = []
        all_predictions = []
        all_probs = []
        all_labels = []
        with tqdm(total=len(all_data.keys()), desc=f"Running LOSO Cross-Validation") as pbar:
            for test_patient in all_data.keys():
                pbar.set_description(f"LOSO - Testing on patient {test_patient}")
                
                X_train_list = [all_data[p].iloc[:,:-1] for p in all_data if p != test_patient]
                y_train_list = [all_data[p].iloc[:,-1] for p in all_data if p != test_patient]
                
                X_train = pd.concat(X_train_list, ignore_index=True)
                y_train = pd.concat(y_train_list, ignore_index=True)
                
                # test set
                X_test = all_data[test_patient].iloc[:,:-1]
                y_test = all_data[test_patient].iloc[:,-1]
                
                # Predict
                scores, y_pred = self.train_and_predict(X_train, y_train, X_test)
                
                all_probs.extend(scores)
                all_predictions.extend(y_pred)
                all_labels.extend(y_test)
                
                # Calculating individual loso stats
                patient_metrics = self._calculate_metrics(y_test, y_pred)
                patient_results.append({
                    'patient_id': test_patient,
                    'sensitivity': patient_metrics['sensitivity'],
                    'specificity': patient_metrics['specificity'],
                    'n_windows': len(y_test),
                    'n_fog_windows': sum(y_test),
                    'y_true': list(y_test),
                    'y_pred': list(y_pred),
                    'y_proba': list(scores)
                })
                
                pbar.update(1)
            
        return all_labels, all_predictions, all_probs, patient_results
            
    # model functions
    def train_and_predict(self, X_train, y_train, X_test, threshold=0.45):
        self.model.fit(X_train, y_train)
        scores = self.model.predict_proba(X_test)[:, 1]
        predictions = (scores >= threshold).astype(int)
        return scores, predictions
    
    def get_feature_importance(self):
        return self.model.feature_importances_
    # model evaluations and data visualization
    def _calculate_metrics(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': (tn, fp, fn, tp)
        }
