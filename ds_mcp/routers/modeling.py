"""Router for modeling endpoints."""

from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


class ModelingRequest(BaseModel):
    """Request model for modeling guidance."""

    task_type: str = Field(
        ..., description="Type of ML task (e.g., classification, regression, clustering)"
    )
    data_size: Optional[str] = Field(
        default="medium", description="Size of the dataset (small, medium, large)"
    )
    data_type: Optional[str] = Field(
        default="tabular", description="Type of data (tabular, text, image, time-series)"
    )
    constraints: Optional[List[str]] = Field(
        default=None,
        description="Constraints (e.g., interpretability, speed, memory)",
    )
    language: Optional[str] = Field(
        default="python", description="Programming language for examples"
    )
    context: Optional[str] = Field(
        default=None, description="Additional context about the modeling task"
    )


class ModelExample(BaseModel):
    """Model for a machine learning model example."""

    name: str = Field(..., description="Name of the model")
    library: str = Field(..., description="Library providing the model")
    description: str = Field(..., description="Description of the model")
    code_example: str = Field(..., description="Example code implementing the model")
    strengths: List[str] = Field(..., description="Strengths of this model")
    weaknesses: List[str] = Field(..., description="Weaknesses of this model")
    use_cases: List[str] = Field(..., description="Appropriate use cases")
    hyperparameters: List[Dict[str, Any]] = Field(
        ..., description="Key hyperparameters to tune"
    )


class ModelingResponse(BaseModel):
    """Response model for modeling guidance."""

    recommended_models: List[ModelExample] = Field(
        ..., description="List of recommended models"
    )
    best_practices: List[str] = Field(..., description="Best practices for modeling")
    workflow_tips: List[str] = Field(..., description="Tips for modeling workflow")


@router.post("/recommendations", response_model=ModelingResponse)
async def get_modeling_recommendations(request: ModelingRequest) -> Dict[str, Any]:
    """Get modeling recommendations based on task and constraints.

    Args:
        request: The modeling guidance request

    Returns:
        Dict containing recommended models, best practices, and workflow tips
    """
    # Classification task with tabular data
    if request.task_type.lower() == "classification" and request.data_type.lower() == "tabular":
        return {
            "recommended_models": [
                {
                    "name": "Gradient Boosting",
                    "library": "lightgbm",
                    "description": "Gradient boosting implementation optimized for efficiency and scalability.",
                    "code_example": """
import lightgbm as lgb
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pl.read_csv("data.csv")
X = df.select(pl.exclude("target")).to_numpy()
y = df["target"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset for LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'binary',  # or 'multiclass' for multi-class classification
    'metric': 'binary_logloss',  # or 'multi_logloss'
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Training
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[valid_data],
    early_stopping_rounds=10
)

# Prediction
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
if params['objective'] == 'binary':
    y_pred_class = [1 if p >= 0.5 else 0 for p in y_pred]
else:
    y_pred_class = y_pred.argmax(axis=1)

# Evaluation
print(classification_report(y_test, y_pred_class))

# Feature importance
importance = model.feature_importance(importance_type='gain')
feature_names = df.select(pl.exclude("target")).columns
for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp}")
""",
                    "strengths": [
                        "High predictive performance",
                        "Handles mixed data types well",
                        "Less prone to overfitting than XGBoost with proper parameters",
                        "Efficient with large datasets",
                        "Good handling of missing values",
                    ],
                    "weaknesses": [
                        "More hyperparameters to tune than simpler models",
                        "Less interpretable than linear models",
                        "Can overfit with small datasets if not properly regularized",
                    ],
                    "use_cases": [
                        "General tabular data classification",
                        "When predictive performance is a priority",
                        "When feature importance is needed",
                    ],
                    "hyperparameters": [
                        {"name": "num_leaves", "description": "Maximum tree leaves for base learners"},
                        {"name": "learning_rate", "description": "Boosting learning rate"},
                        {"name": "n_estimators", "description": "Number of boosted trees"},
                        {"name": "feature_fraction", "description": "Fraction of features to use in each iteration"},
                        {"name": "bagging_fraction", "description": "Fraction of data to use for each iteration"},
                    ],
                },
                {
                    "name": "Random Forest",
                    "library": "scikit-learn",
                    "description": "Ensemble of decision trees trained on different subsets of data and features.",
                    "code_example": """
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import polars as pl
import numpy as np

# Load data
df = pl.read_csv("data.csv")
X = df.select(pl.exclude("target")).to_numpy()
y = df["target"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Hyperparameter tuning (optional)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)
best_rf = rf_random.best_estimator_

# Alternatively, just fit the base model
# rf.fit(X_train, y_train)
# best_rf = rf

# Predictions
y_pred = best_rf.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Feature importance
importance = best_rf.feature_importances_
feature_names = df.select(pl.exclude("target")).columns
for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {imp}")
""",
                    "strengths": [
                        "Robust to overfitting",
                        "Handles high-dimensional data well",
                        "Can handle mixed data types",
                        "Provides feature importance",
                        "Works well out-of-the-box with minimal tuning",
                    ],
                    "weaknesses": [
                        "Can be memory-intensive for large datasets",
                        "Less interpretable than linear models or single decision trees",
                        "May not perform as well as gradient boosting for complex relationships",
                    ],
                    "use_cases": [
                        "General tabular data classification",
                        "When robustness is a priority",
                        "When you need a good baseline with minimal tuning",
                    ],
                    "hyperparameters": [
                        {"name": "n_estimators", "description": "Number of trees in the forest"},
                        {"name": "max_depth", "description": "Maximum depth of each tree"},
                        {"name": "min_samples_split", "description": "Minimum samples required to split a node"},
                        {"name": "min_samples_leaf", "description": "Minimum samples required at a leaf node"},
                        {"name": "max_features", "description": "Number of features to consider for best split"},
                    ],
                },
            ],
            "best_practices": [
                "Always split your data into training, validation, and test sets",
                "Use cross-validation for more reliable performance estimates",
                "Start with simpler models as baselines before using more complex ones",
                "Monitor for overfitting, especially with complex models",
                "Consider model interpretability requirements early in the process",
                "Document your modeling decisions and results",
                "Version your models and datasets",
            ],
            "workflow_tips": [
                "Implement a consistent preprocessing pipeline for both training and inference",
                "Use hyperparameter optimization but be wary of overfitting",
                "Consider feature selection to improve model performance and efficiency",
                "For imbalanced datasets, use appropriate techniques (sampling, class weights, or specialized metrics)",
                "Implement early stopping to prevent overfitting during training",
                "Save models with pickle, joblib, or ONNX for deployment",
            ],
        }
    
    # Regression task
    elif request.task_type.lower() == "regression":
        return {
            "recommended_models": [
                {
                    "name": "ElasticNet",
                    "library": "scikit-learn",
                    "description": "Linear regression with combined L1 and L2 regularization.",
                    "code_example": """
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pl.read_csv("data.csv")
X = df.select(pl.exclude("target")).to_numpy()
y = df["target"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elasticnet', ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
        alphas=np.logspace(-4, 0, 20),
        cv=5,
        max_iter=10000
    ))
])

# Fit model
pipeline.fit(X_train, y_train)

# Get the best model
best_alpha = pipeline.named_steps['elasticnet'].alpha_
best_l1_ratio = pipeline.named_steps['elasticnet'].l1_ratio_
print(f"Best alpha: {best_alpha}")
print(f"Best l1_ratio: {best_l1_ratio}")

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Feature importance (coefficients)
feature_names = df.select(pl.exclude("target")).columns
coefs = pipeline.named_steps['elasticnet'].coef_
important_features = [(name, coef) for name, coef in zip(feature_names, coefs) if coef != 0]
for name, coef in sorted(important_features, key=lambda x: abs(x[1]), reverse=True):
    print(f"{name}: {coef:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.show()
""",
                    "strengths": [
                        "Built-in feature selection via L1 regularization",
                        "Handles multicollinearity well via L2 regularization",
                        "Highly interpretable through coefficients",
                        "Works well with high-dimensional data",
                        "Efficient training and inference",
                    ],
                    "weaknesses": [
                        "May underperform for highly non-linear relationships",
                        "Sensitive to feature scaling (requires standardization)",
                        "May require more feature engineering than tree-based models",
                    ],
                    "use_cases": [
                        "Linear regression problems",
                        "When feature selection is important",
                        "When interpretability is required",
                        "High-dimensional datasets",
                    ],
                    "hyperparameters": [
                        {"name": "alpha", "description": "Regularization strength"},
                        {"name": "l1_ratio", "description": "Balance between L1 and L2 regularization (0 = Ridge, 1 = Lasso)"},
                        {"name": "max_iter", "description": "Maximum number of iterations"},
                        {"name": "tol", "description": "Tolerance for stopping criteria"},
                    ],
                },
            ],
            "best_practices": [
                "Scale features for linear models",
                "Check assumptions of linear regression (if using linear models)",
                "Evaluate models on multiple metrics, not just one",
                "Consider log-transforming the target for skewed distributions",
                "Address outliers appropriately for your use case",
            ],
            "workflow_tips": [
                "Start with a simple linear model as a baseline",
                "Analyze residuals to identify patterns of error",
                "For time series data, be careful with data leakage during preprocessing",
                "Consider ensembling multiple models for better performance",
                "Plot actual vs predicted values to visually evaluate performance",
            ],
        }
    
    # Default response
    else:
        return {
            "recommended_models": [
                {
                    "name": "Random Forest",
                    "library": "scikit-learn",
                    "description": "Versatile ensemble method that works well across many problem types.",
                    "code_example": """
from sklearn.ensemble import RandomForestClassifier  # or RandomForestRegressor for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # or regression metrics
import polars as pl

# Load data
df = pl.read_csv("data.csv")
X = df.select(pl.exclude("target")).to_numpy()
y = df["target"].to_numpy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
""",
                    "strengths": [
                        "Works well across many problems",
                        "Requires minimal preprocessing",
                        "Provides feature importance",
                        "Resistant to overfitting",
                    ],
                    "weaknesses": [
                        "Can be computationally expensive",
                        "Less interpretable than linear models",
                    ],
                    "use_cases": [
                        "General-purpose classifier or regressor",
                        "When you need a strong baseline",
                    ],
                    "hyperparameters": [
                        {"name": "n_estimators", "description": "Number of trees"},
                        {"name": "max_depth", "description": "Maximum depth of trees"},
                        {"name": "min_samples_split", "description": "Minimum samples to split a node"},
                    ],
                },
            ],
            "best_practices": [
                "Always split your data into training and test sets",
                "Use cross-validation for hyperparameter tuning",
                "Start with simple models before complex ones",
                "Document your modeling process",
            ],
            "workflow_tips": [
                "Implement a consistent preprocessing pipeline",
                "Save and version your models",
                "Monitor model performance over time",
                "Consider model interpretability requirements",
            ],
        }
