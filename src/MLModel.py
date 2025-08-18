from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
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
        all_predictions = []
        all_labels = []
        for test_patient in all_data.keys():
            print(f"Testing on patient {test_patient}")
            X_train_list = [all_data[p].iloc[:,:-1] for p in all_data if p != test_patient]
            y_train_list = [all_data[p].iloc[:,-1] for p in all_data if p != test_patient]
            
            X_train = pd.concat(X_train_list, ignore_index=True)
            y_train = pd.concat(y_train_list, ignore_index=True)
            # test set
            X_test = all_data[test_patient].iloc[:,:-1]
            y_test = all_data[test_patient].iloc[:,-1]
            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
            # Predict
            y_pred = self.train_and_predict(X_train, y_train, X_test)
            
            all_predictions.extend(y_pred)
            all_labels.extend(y_test)
            
        return all_labels, all_predictions
            
    # model functions
    def train_and_predict(self, X_train, y_train, X_test, threshold=0.43):
        self.model.fit(X_train, y_train)
        scores = self.model.predict_proba(X_test)[:, 1]
        return (scores >= threshold).astype(int)
        
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

class ModelVisuals:
    def plot_confusion_matrix():
        pass
    def eval_loso():
        pass
    def plot_feature_importance():
        pass
    def plot_threshold_performance():
        pass