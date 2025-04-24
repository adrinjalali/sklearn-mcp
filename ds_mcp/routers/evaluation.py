"""Router for model evaluation endpoints."""

from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


class EvaluationRequest(BaseModel):
    """Request model for evaluation guidance."""

    task_type: str = Field(
        ..., description="Type of ML task (e.g., classification, regression, clustering)"
    )
    model_type: Optional[str] = Field(
        default=None, description="Type of model being evaluated"
    )
    data_type: Optional[str] = Field(
        default="tabular", description="Type of data (tabular, text, image, time-series)"
    )
    context: Optional[str] = Field(
        default=None, description="Additional context about the evaluation task"
    )


class MetricExample(BaseModel):
    """Model for an evaluation metric example."""

    name: str = Field(..., description="Name of the metric")
    description: str = Field(..., description="Description of the metric")
    code_example: str = Field(..., description="Example code implementing the metric")
    use_cases: List[str] = Field(..., description="Appropriate use cases for this metric")
    interpretation: str = Field(
        ..., description="How to interpret the metric values"
    )


class EvaluationResponse(BaseModel):
    """Response model for evaluation guidance."""

    metrics: List[MetricExample] = Field(
        ..., description="List of relevant evaluation metrics"
    )
    visualization_techniques: List[Dict[str, Any]] = Field(
        ..., description="Techniques for visualizing model performance"
    )
    best_practices: List[str] = Field(..., description="Best practices for model evaluation")


@router.post("/metrics", response_model=EvaluationResponse)
async def get_evaluation_guidance(request: EvaluationRequest) -> Dict[str, Any]:
    """Get guidance on evaluation metrics and techniques.

    Args:
        request: The evaluation guidance request

    Returns:
        Dict containing metrics, visualization techniques, and best practices
    """
    # Binary classification task
    if request.task_type.lower() == "classification":
        is_binary = True  # Default to binary, could be determined from context in the future
        
        metrics = [
            {
                "name": "ROC AUC",
                "description": "Area Under the Receiver Operating Characteristic Curve. Measures the ability of the model to distinguish between classes across various thresholds.",
                "code_example": """
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import polars as pl
import numpy as np

# Assuming y_test is ground truth and y_pred_proba is predicted probabilities
y_test = np.array([0, 1, 0, 1, 1, 0])
y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
""",
                "use_cases": [
                    "Binary classification problems",
                    "When class balance is uneven",
                    "When ranking quality is important",
                ],
                "interpretation": "Values range from 0.5 (random) to 1.0 (perfect). Above 0.8 is generally considered good, above 0.9 excellent.",
            },
            {
                "name": "Precision-Recall",
                "description": "Metrics focused on the positive class. Precision measures the accuracy of positive predictions, while recall measures the ability to find all positive instances.",
                "code_example": """
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Assuming y_test is ground truth, y_pred is class predictions, 
# and y_pred_proba is predicted probabilities
y_test = np.array([0, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])
y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.7, 0.6])

# Calculate metrics at threshold
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Calculate Precision-Recall curve
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label=f'AP = {average_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
""",
                "use_cases": [
                    "When false positives and false negatives have different costs",
                    "Imbalanced datasets where the positive class is the minority",
                    "When focus is on the positive class performance",
                ],
                "interpretation": "Precision and recall range from 0 to 1. Higher values are better. F1 score is the harmonic mean of precision and recall, also ranging from 0 to 1.",
            },
            {
                "name": "Confusion Matrix",
                "description": "A table showing the counts of true positives, false positives, true negatives, and false negatives.",
                "code_example": """
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Assuming y_test is ground truth and y_pred is predicted classes
y_test = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 0])

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix with labels
class_names = ['Negative', 'Positive']  # Customize based on your classes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()
print(f"True Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

# Calculate additional metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall/Sensitivity: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
""",
                "use_cases": [
                    "Any classification problem",
                    "When you need a detailed breakdown of error types",
                    "When different types of errors have different implications",
                ],
                "interpretation": "Diagonal elements represent correct predictions. Off-diagonal elements represent errors. The goal is to maximize the diagonal and minimize the off-diagonal elements.",
            },
        ]

        visualizations = [
            {
                "name": "ROC Curve",
                "description": "Plots the True Positive Rate against the False Positive Rate at various threshold settings.",
                "use_cases": ["Binary classification", "Threshold selection"],
            },
            {
                "name": "Precision-Recall Curve",
                "description": "Plots Precision against Recall at various threshold settings.",
                "use_cases": ["Imbalanced classification", "When focus is on positive class"],
            },
            {
                "name": "Calibration Plot",
                "description": "Shows how well predicted probabilities match observed frequencies.",
                "use_cases": ["When probability estimates need to be reliable"],
            },
        ]

        best_practices = [
            "Always evaluate on a separate test set not used during training",
            "Use cross-validation for more reliable performance estimates",
            "Consider the business impact of different types of errors",
            "For imbalanced datasets, don't rely solely on accuracy",
            "Ensure your evaluation metrics align with the business objectives",
            "Consider the confidence of predictions, not just the predicted class",
            "Perform stratified sampling to maintain class distributions in train/test splits",
            "Report multiple metrics for a more complete picture of model performance",
        ]

        return {
            "metrics": metrics,
            "visualization_techniques": visualizations,
            "best_practices": best_practices,
        }
    
    # Regression task
    elif request.task_type.lower() == "regression":
        metrics = [
            {
                "name": "Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)",
                "description": "Measures the average squared difference between predicted and actual values. RMSE is the square root of MSE and has the same units as the target variable.",
                "code_example": """
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Assuming y_test is ground truth and y_pred is predictions
y_test = np.array([3.1, 4.2, 5.3, 2.5, 6.0])
y_pred = np.array([2.9, 4.0, 5.0, 2.0, 5.8])

# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Visualize residuals
residuals = y_test - y_pred

plt.figure(figsize=(12, 5))

# Plot 1: Actual vs Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.grid(True)

# Plot 2: Residuals
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)

plt.tight_layout()
plt.show()
""",
                "use_cases": [
                    "General regression problems",
                    "When larger errors should be penalized more heavily",
                    "When the scale of the error matters",
                ],
                "interpretation": "Lower values are better. RMSE is in the same units as the target variable, making it more interpretable than MSE.",
            },
            {
                "name": "Mean Absolute Error (MAE)",
                "description": "Measures the average absolute difference between predicted and actual values.",
                "code_example": """
from sklearn.metrics import mean_absolute_error
import numpy as np

# Assuming y_test is ground truth and y_pred is predictions
y_test = np.array([3.1, 4.2, 5.3, 2.5, 6.0])
y_pred = np.array([2.9, 4.0, 5.0, 2.0, 5.8])

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

print(f"MAE: {mae:.4f}")
""",
                "use_cases": [
                    "When all errors should be penalized equally",
                    "When outliers should have less influence than with MSE",
                    "When interpretability is important",
                ],
                "interpretation": "Lower values are better. MAE is in the same units as the target variable and represents the average absolute error.",
            },
            {
                "name": "R-squared (Coefficient of Determination)",
                "description": "Measures the proportion of variance in the dependent variable that is predictable from the independent variables.",
                "code_example": """
from sklearn.metrics import r2_score
import numpy as np

# Assuming y_test is ground truth and y_pred is predictions
y_test = np.array([3.1, 4.2, 5.3, 2.5, 6.0])
y_pred = np.array([2.9, 4.0, 5.0, 2.0, 5.8])

# Calculate R-squared
r2 = r2_score(y_test, y_pred)

print(f"R-squared: {r2:.4f}")
""",
                "use_cases": [
                    "General regression problems",
                    "When you want to measure the proportion of variance explained by the model",
                    "When comparing different models",
                ],
                "interpretation": "Ranges from 0 to 1, with higher values being better. A value of 0 means the model provides no better predictions than the mean of the target, while 1 means perfect predictions.",
            },
        ]

        visualizations = [
            {
                "name": "Actual vs Predicted Plot",
                "description": "Scatter plot of actual values against predicted values.",
                "use_cases": ["Visualizing overall model fit", "Identifying systematic errors"],
            },
            {
                "name": "Residual Plot",
                "description": "Scatter plot of residuals (actual - predicted) against predicted values.",
                "use_cases": ["Checking for heteroscedasticity", "Identifying non-linear patterns"],
            },
            {
                "name": "Residual Distribution",
                "description": "Histogram or density plot of residuals.",
                "use_cases": ["Checking for normality of residuals", "Identifying bias"],
            },
        ]

        best_practices = [
            "Always evaluate on a separate test set not used during training",
            "Use cross-validation for more reliable performance estimates",
            "Consider the business context when selecting evaluation metrics",
            "Check for heteroscedasticity and non-linear patterns in residuals",
            "Compare your model to simple baselines like mean or median prediction",
            "Consider transforming the target variable if the error distribution is skewed",
            "Report multiple metrics for a more complete picture of model performance",
            "Use confidence intervals to quantify prediction uncertainty",
        ]

        return {
            "metrics": metrics,
            "visualization_techniques": visualizations,
            "best_practices": best_practices,
        }
    
    # Default response
    else:
        return {
            "metrics": [
                {
                    "name": "Task-appropriate metric",
                    "description": "Choose metrics based on your specific task and business objectives.",
                    "code_example": """
# For classification
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# For regression
from sklearn.metrics import mean_squared_error, r2_score

# Calculate appropriate metrics for your task
""",
                    "use_cases": ["Any machine learning task"],
                    "interpretation": "Interpretation depends on the specific metric chosen.",
                }
            ],
            "visualization_techniques": [
                {
                    "name": "Performance visualization",
                    "description": "Visualize model performance using task-appropriate plots.",
                    "use_cases": ["Any machine learning task"],
                }
            ],
            "best_practices": [
                "Clearly define your evaluation criteria before model development",
                "Always evaluate on a separate test set not used during training",
                "Use cross-validation for more reliable performance estimates",
                "Select metrics aligned with business objectives",
                "Compare against appropriate baselines",
            ],
        }
