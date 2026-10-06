import numpy as np
from sklearn.model_selection import cross_val_score
from typing import Any

def run_cross_validation(model: Any, X: Any, y: Any, cv: int = 5) -> str:
    """
    Enforces strict 5-fold cross-validation rules and returns non-hallucinable scores.
    """
    print(f"🔄 [EVALUATION PROTOCOL] Running {cv}-fold Cross-Validation...")
    
    # Run the CV
    scores = cross_val_score(model, X, y, cv=cv)
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    result = f"Model: {model.__class__.__name__} | Mean CV Score: {mean_score:.4f} ± {std_score:.4f} (folds: {cv})"
    print(result)
    return result
