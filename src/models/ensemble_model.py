import xgboost as xgb
import numpy as np
import joblib
from typing import Tuple

class PPIEnsemble:
    def __init__(self, meta_model_path: str = None):
        """
        Ensemble model using Stacking with 4 features.
        Features: [seq_prob, gat_prob, |seq_prob - 0.5|, |gat_prob - 0.5|]
        """
        self.meta_model = None
        if meta_model_path:
            try:
                self.meta_model = joblib.load(meta_model_path)
                print(f"Loaded meta-learner from {meta_model_path}")
            except:
                print("Could not load meta-learner. Path might be invalid or not exist yet.")

    @staticmethod
    def _build_features(base_preds_1: np.ndarray, base_preds_2: np.ndarray) -> np.ndarray:
        """
        Build feature matrix for the meta-learner.
        Adds |p - 0.5| confidence features for models.
        """
        conf_1 = np.abs(base_preds_1 - 0.5)
        conf_2 = np.abs(base_preds_2 - 0.5)
        
        return np.column_stack((base_preds_1, base_preds_2, conf_1, conf_2))

    def train_stacking(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, labels: np.ndarray):
        """
        Trains the XGBoost meta-learner.
        args:
            base_preds_1: Predictions from Sequence Model (N,)
            base_preds_2: Predictions from Graph Model (N,)
            labels: True labels (N,)
        """
        X = self._build_features(base_preds_1, base_preds_2)
        
        print(f"Training XGBoost Meta-Learner with {X.shape[1]} features...")
        self.meta_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
        self.meta_model.fit(X, labels)
        print("Meta-learner training complete.")

    def predict(self, base_preds_1: np.ndarray, base_preds_2: np.ndarray, method: str = "stacking") -> np.ndarray:
        """
        Predicts final probability.
        method: 'stacking' or 'soft_voting'
        """
        if method == "soft_voting":
            return (base_preds_1 + base_preds_2) / 2.0
            
        elif method == "stacking":
            if self.meta_model is None:
                raise ValueError("Meta-learner not trained/loaded.")
            
            X = self._build_features(base_preds_1, base_preds_2)
            # Predict probabilities
            return self.meta_model.predict_proba(X)[:, 1]
            
        else:
            raise ValueError(f"Unknown method: {method}")

    def save(self, path: str):
        if self.meta_model:
            joblib.dump(self.meta_model, path)
            print(f"Meta-learner saved to {path}")
