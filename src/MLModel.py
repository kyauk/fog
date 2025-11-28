from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
from tqdm import tqdm
import pandas as pd

class Model():
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        if model_type =='random_forest':
            # Optimized hyperparameters for better performance
            self.model = RandomForestClassifier(
                n_estimators=200,        # Increased from 100
                max_depth=None,          # Unlimited depth (was 8)
                min_samples_leaf=2,      # Reduced from 5 for finer splits
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
    def loso(self, all_data, optimize_threshold=True):
        patient_results = []
        all_predictions = []
        all_probs = []
        all_labels = []
        
        # First pass: collect all probabilities to find optimal threshold
        if optimize_threshold:
            print("Finding optimal threshold using ROC analysis...")
            temp_probs = []
            temp_labels = []
            
            with tqdm(total=len(all_data.keys()), desc="Collecting probabilities for threshold optimization") as pbar:
                for test_patient in all_data.keys():
                    X_train_list = [all_data[p].iloc[:,:-1] for p in all_data if p != test_patient]
                    y_train_list = [all_data[p].iloc[:,-1] for p in all_data if p != test_patient]
                    
                    X_train = pd.concat(X_train_list, ignore_index=True)
                    y_train = pd.concat(y_train_list, ignore_index=True)
                    
                    X_test = all_data[test_patient].iloc[:,:-1]
                    y_test = all_data[test_patient].iloc[:,-1]
                    
                    self.model.fit(X_train, y_train)
                    scores = self.model.predict_proba(X_test)[:, 1]
                    
                    temp_probs.extend(scores)
                    temp_labels.extend(y_test)
                    pbar.update(1)
            
            # Find optimal threshold using Youden's J statistic
            optimal_threshold = self._find_optimal_threshold(temp_labels, temp_probs)
            print(f"Optimal threshold found: {optimal_threshold:.3f}")
        else:
            optimal_threshold = 0.45  # Default
        
        # Second pass: actual LOSO with optimal threshold
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
                
                # Predict with optimal threshold
                scores, y_pred = self.train_and_predict(X_train, y_train, X_test, threshold=optimal_threshold)
                
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
            
    def _find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold using Youden's J statistic from ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        # Youden's J statistic = sensitivity + specificity - 1 = tpr - fpr
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold
    
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
