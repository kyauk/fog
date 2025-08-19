from .preprocess import DataPreprocessor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from tqdm import tqdm
# data visualization imports
from sklearn.metrics import confusion_matrix,roc_curve, auc, classification_report, precision_recall_curve
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DataVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    def plot_raw_vs_processed_data(self, patient_file, preprocessor: DataPreprocessor, start_idx=0, duration= 30):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        # Load data
        data = np.genfromtxt(patient_file, delimiter=',', skip_header=1)
        thigh_data = data[:,4]
        labels = data[:,-1]
        # Create a time array
        time = np.arange(len(thigh_data)) / preprocessor.sampling_rate
        # Select window to display
        display_samples = int(duration * preprocessor.sampling_rate)
        start_idx = start_idx
        end_idx = min(start_idx + display_samples, len(thigh_data))
        
        time_window = time[start_idx:end_idx]
        raw_window = thigh_data[start_idx:end_idx]
        labels_window = labels[start_idx:end_idx]
        # Apply Preprocessing
        filtered_data = preprocessor.apply_butter(thigh_data)
        filtered_window = filtered_data[start_idx:end_idx]
        # Plot Raw Signals
        ax1 = axes[0]
        ax1.plot(time_window, raw_window, 'b-', alpha=0.7, linewidth=0.8, label='Raw Signal')
        # Highlight FoG Events
        fog_mask = labels_window == 2
        if np.any(fog_mask):
            ax1.fill_between(time_window, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                            where=fog_mask, alpha=0.3, color='red', label='FoG Event')
        ax1.set_ylabel('Acceleration (mg)')
        ax1.set_title('Raw Signal with FoG Events', fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
         # Plot Preprocessed Signals
        ax2 = axes[1]
        ax2.plot(time_window, filtered_window, 'g-', alpha=0.7, linewidth=0.8, label='Filtered Signal')
        
        # Highlight FoG events
        if np.any(fog_mask):
            ax2.fill_between(time_window, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                            where=fog_mask, alpha=0.3, color='red', label='FoG Event')
        
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (mg)')
        ax2.set_title('Preprocessed Signal (Butterworth Filtered)', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Signal Comparison (Window starting at {start_idx/preprocessor.sampling_rate:.1f}s)', 
                fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    
    def plot_confusion_matrix_roc_pr(self, y_true, y_pred, y_proba=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # making confusion matrix
        ax1 = axes[0]
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                   xticklabels=['No FoG', 'FoG'], yticklabels=['No FoG', 'FoG'], ax=ax1)
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        ax1.set_title('Confusion Matrix', fontweight='bold')
        
        # calculate metrics
        
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics_text = f'Sensitivity: {sensitivity:.1%}\nSpecificity: {specificity:.1%}\nAccuracy: {accuracy:.1%}'
        ax1.text(2.5, 0.5, metrics_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # roc curve
        
        ax2 = axes[1]
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC Curve', fontweight='bold')
            ax2.legend(loc='lower right')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No probability scores available\nfor ROC curve', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('ROC Curve (Not Available)', fontweight='bold')
        
        # pr curve
        ax3 = axes[2]
        if y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ax3.plot(recall, precision, 'g-', linewidth=2)
            ax3.set_xlabel('Recall (Sensitivity)')
            ax3.set_ylabel('Precision')
            ax3.set_title('Precision-Recall Curve', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # add avg precision score
            from sklearn.metrics import average_precision_score
            avg_precision = average_precision_score(y_true, y_proba)
            ax3.text(0.1, 0.1, f'Avg Precision: {avg_precision:.3f}', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, 'No probability scores available\nfor PR curve', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Precision-Recall Curve (Not Available)', fontweight='bold')
        
        plt.suptitle('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    