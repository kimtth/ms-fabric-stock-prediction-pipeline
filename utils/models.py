"""
Model Training and Prediction Utility
Handles ML model training, evaluation, and prediction for stock classification
"""

import xgboost as xgb
import joblib
import json
import pandas as pd
import numpy as np
import warnings

from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings('ignore')


class StockClassifier:
    """Train and evaluate classification models for stock trading signals"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize classifier with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.feature_importance = None
    
    def prepare_features(
        self, 
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare feature matrix for training
        
        Args:
            df: DataFrame with indicators and target
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (prepared DataFrame, feature column names)
        """
        df = df.copy()
        
        # Default exclusions
        if exclude_cols is None:
            exclude_cols = [
                'date', 'Ticker', 'FetchTimestamp',
                'open', 'high', 'low', 'close', 'volume',
                'dividends', 'stock_splits', 'year', 'month',
                'target', 'future_return'
            ]
        
        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Remove rows with NaN in features or target
        df_clean = df[feature_cols + ['target']].dropna()
        
        print(f"✓ Prepared {len(feature_cols)} features")
        print(f"  Total samples: {len(df_clean)}")
        print(f"  Removed {len(df) - len(df_clean)} rows with missing values")
        
        self.feature_columns = feature_cols
        
        return df_clean, feature_cols
    
    def split_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets (time-aware)
        
        Args:
            df: Prepared DataFrame
            feature_cols: List of feature column names
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Use time-based split (not random) for time series
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]
        y_train = train_df['target']
        y_test = test_df['target']
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        print("✓ Data split:")
        print(f"  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
        print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict] = None,
        use_smote: bool = True
    ):
        """
        Train XGBoost classifier with class balancing
        
        Args:
            X_train: Training features
            y_train: Training labels (-1, 0, 1)
            hyperparameters: Optional hyperparameters
            use_smote: Whether to use SMOTE for oversampling minority classes
        """
        print("\nTraining XGBoost Classifier...")
        
        # XGBoost requires classes to be 0-indexed (0, 1, 2)
        # Map -1 -> 0, 0 -> 1, 1 -> 2
        self.label_mapping = {-1: 0, 0: 1, 1: 2}
        self.inverse_label_mapping = {0: -1, 1: 0, 2: 1}
        y_train_mapped = y_train.map(self.label_mapping)
        
        # Calculate class weights to handle imbalance
        class_counts = y_train_mapped.value_counts().sort_index()
        total_samples = len(y_train_mapped)
        class_weights = {cls: total_samples / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
        
        print(f"  Original class distribution: {class_counts.to_dict()}")
        print(f"  Calculated class weights: {class_weights}")
        
        # Apply SMOTE if enabled
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42, k_neighbors=3)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_mapped)
                print(f"  ✓ SMOTE applied: {len(X_train)} → {len(X_train_balanced)} samples")
                print(f"  Balanced class distribution: {pd.Series(y_train_balanced).value_counts().sort_index().to_dict()}")
                X_train = X_train_balanced
                y_train_mapped = y_train_balanced
            except ImportError:
                print("  ⚠ imbalanced-learn not installed, skipping SMOTE")
                print("  Install with: pip install imbalanced-learn")
        
        if hyperparameters is None:
            hyperparameters = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softmax',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42
            }
        
        # Add class weights to model with aggressive scaling for minority classes
        # Scale up minority class weights by 3x for better detection
        minority_boost = 3.0
        adjusted_weights = class_weights.copy()
        for cls in adjusted_weights:
            if class_counts[cls] < total_samples / len(class_counts):  # Minority class
                adjusted_weights[cls] *= minority_boost
        
        sample_weights = np.array([adjusted_weights[label] for label in y_train_mapped])
        print(f"  Adjusted class weights (with {minority_boost}x boost): {adjusted_weights}")
        
        self.model = xgb.XGBClassifier(**hyperparameters)
        self.model.fit(X_train, y_train_mapped, sample_weight=sample_weights)
        
        print("✓ XGBoost training completed with aggressive class balancing")
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hyperparameters: Optional[Dict] = None
    ):
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameters: Optional hyperparameters
        """
        print("\nTraining Random Forest Classifier...")
        
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.model = RandomForestClassifier(**hyperparameters)
        self.model.fit(X_train, y_train)
        
        print("✓ Random Forest training completed")
    
    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        label_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels (-1, 0, 1)
            label_names: Optional label names for display
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        if label_names is None:
            label_names = ['Sell', 'Hold', 'Buy']
        
        # Map test labels if using XGBoost
        if hasattr(self, 'label_mapping'):
            y_test_mapped = y_test.map(self.label_mapping)
        else:
            y_test_mapped = y_test
        
        # Predictions
        y_pred_mapped = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Map predictions back to original labels if needed
        if hasattr(self, 'inverse_label_mapping'):
            y_pred = pd.Series(y_pred_mapped).map(self.inverse_label_mapping)
        else:
            y_pred = y_pred_mapped
        
        # Metrics (use mapped values for calculation)
        accuracy = accuracy_score(y_test_mapped, y_pred_mapped)
        precision = precision_score(y_test_mapped, y_pred_mapped, average='weighted', zero_division=0)
        recall = recall_score(y_test_mapped, y_pred_mapped, average='weighted', zero_division=0)
        f1 = f1_score(y_test_mapped, y_pred_mapped, average='weighted', zero_division=0)
        
        # Multi-class ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test_mapped, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:  
            print(f"✗ Error in model evaluation: {str(e)}")
            roc_auc = None
        
        # Classification report (use mapped values)
        class_report = classification_report(
            y_test_mapped, y_pred_mapped,
            target_names=label_names,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_imp = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance = feature_imp
        
        # Print results
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC:   {roc_auc:.4f}")
        print("\nClassification Report:")
        print(class_report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("="*60 + "\n")
        
        if self.feature_importance is not None:
            print("Top 10 Most Important Features:")
            print(self.feature_importance.head(10).to_string(index=False))
            print()
        
        # Return metrics dictionary
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Tuple of (predictions in original labels -1/0/1, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale features
        X_scaled = self.scaler.transform(X[self.feature_columns])
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        # Predict
        predictions_mapped = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Map predictions back to original labels if using XGBoost
        if hasattr(self, 'inverse_label_mapping'):
            predictions = np.array([self.inverse_label_mapping[p] for p in predictions_mapped])
        else:
            predictions = predictions_mapped
        
        return predictions, probabilities
    
    def save_model(self, filepath: str):
        """Save model and scaler to disk"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'label_mapping': getattr(self, 'label_mapping', None),
            'inverse_label_mapping': getattr(self, 'inverse_label_mapping', None)
        }
        
        joblib.dump(model_artifacts, filepath)
        print(f"✓ Model saved to {Path(filepath).name}")
    
    def load_model(self, filepath: str):
        """Load model and scaler from disk"""
        model_artifacts = joblib.load(filepath)
        
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.feature_columns = model_artifacts['feature_columns']
        
        if model_artifacts['feature_importance'] is not None:
            self.feature_importance = pd.DataFrame(model_artifacts['feature_importance'])
        
        # Load label mappings if they exist
        self.label_mapping = model_artifacts.get('label_mapping')
        self.inverse_label_mapping = model_artifacts.get('inverse_label_mapping')
        
        print(f"✓ Model loaded from {Path(filepath).name}")
    
    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str = 'xgboost',
        cv_folds: int = 5
    ) -> Dict:
        """
        Perform hyperparameter tuning with GridSearchCV
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: 'xgboost' or 'random_forest'
            cv_folds: Number of cross-validation folds
            
        Returns:
            Best hyperparameters
        """
        print(f"\nPerforming hyperparameter tuning for {model_type}...")
        
        if model_type == 'xgboost':
            base_model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                random_state=42
            )
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:  # random_forest
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
