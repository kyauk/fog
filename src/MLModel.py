from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
    # model functions
    def train_and_predict(self, X_train, y_train, X_test, threshold=0.5):
        self.model.fit(X_train, y_train)
        scores = self.model.predict_proba(X_test)[:, 1]
        return (scores >= threshold).astype(int)
        
    def get_feature_importance(self):
        return self.model.feature_importances_
    # model evaluations and data visualization
    def _calculate_metrics(self, y_true, y_pred):
        pass

class ModelVisuals:
    def plot_confusion_matrix():
        pass
    def eval_loso():
        pass
    def plot_feature_importance():
        pass
    def plot_threshold_performance():
        pass