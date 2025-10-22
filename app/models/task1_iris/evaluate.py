import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
import seaborn as sns
import os

class IrisModelEvaluator:
    """
    Evaluator for Iris Classification Model
    Calculates accuracy, precision, recall, and generates visualizations
    """
    
    def __init__(self, label_mapping=None):
        self.label_mapping = label_mapping
        self.class_names = None
        if label_mapping:
            self.class_names = list(label_mapping.keys())
    
    def evaluate(self, model, X_test, y_test):
        """
        Comprehensive evaluation of the model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary with all metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print results
        self.print_evaluation_report(metrics)
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision (macro and weighted averages)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        
        # Recall (macro and weighted averages)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        
        # F1 Score
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        # Confusion Matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification Report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        return metrics
    
    def print_evaluation_report(self, metrics):
        """
        Print formatted evaluation report
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        
        print("\n--- Precision ---")
        print(f"  Macro Average: {metrics['precision_macro']:.4f}")
        print(f"  Weighted Average: {metrics['precision_weighted']:.4f}")
        if self.class_names:
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}: {metrics['precision_per_class'][i]:.4f}")
        
        print("\n--- Recall ---")
        print(f"  Macro Average: {metrics['recall_macro']:.4f}")
        print(f"  Weighted Average: {metrics['recall_weighted']:.4f}")
        if self.class_names:
            for i, class_name in enumerate(self.class_names):
                print(f"  {class_name}: {metrics['recall_per_class'][i]:.4f}")
        
        print("\n--- F1 Score ---")
        print(f"  Macro Average: {metrics['f1_macro']:.4f}")
        print(f"  Weighted Average: {metrics['f1_weighted']:.4f}")
        
        print("\n--- Confusion Matrix ---")
        print(metrics['confusion_matrix'])
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(self, confusion_mat, save_path=None):
        """
        Plot confusion matrix as heatmap
        
        Args:
            confusion_mat: Confusion matrix
            save_path: Path to save the figure
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_mat, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.close()
        return save_path
    
    def plot_feature_importance(self, feature_importance, save_path=None):
        """
        Plot feature importance
        
        Args:
            feature_importance: Dictionary of feature importance
            save_path: Path to save the figure
        """
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette('viridis', len(features))
        plt.barh(features, importances, color=colors)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature Importance in Decision Tree', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.close()
        return save_path
    
    def plot_metrics_comparison(self, metrics, save_path=None):
        """
        Plot comparison of precision, recall, and F1 score per class
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save the figure
        """
        if not self.class_names:
            return None
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        rects1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', color='skyblue')
        rects2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall', color='lightcoral')
        rects3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1 Score', color='lightgreen')
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Metrics per Class', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics comparison plot saved to: {save_path}")
        
        plt.close()
        return save_path
    
    def generate_all_plots(self, metrics, feature_importance, output_dir):
        """
        Generate all evaluation plots
        
        Args:
            metrics: Dictionary of metrics
            feature_importance: Feature importance dictionary
            output_dir: Directory to save plots
            
        Returns:
            Dictionary with paths to all plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = {}
        
        # Confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_paths['confusion_matrix'] = self.plot_confusion_matrix(
            metrics['confusion_matrix'], cm_path
        )
        
        # Feature importance
        fi_path = os.path.join(output_dir, 'feature_importance.png')
        plot_paths['feature_importance'] = self.plot_feature_importance(
            feature_importance, fi_path
        )
        
        # Metrics comparison
        mc_path = os.path.join(output_dir, 'metrics_comparison.png')
        plot_paths['metrics_comparison'] = self.plot_metrics_comparison(
            metrics, mc_path
        )
        
        return plot_paths