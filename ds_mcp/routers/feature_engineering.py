"""Router for feature engineering endpoints."""

from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


class FeatureEngineeringRequest(BaseModel):
    """Request model for feature engineering guidance."""

    data_type: str = Field(..., description="Type of data (e.g., tabular, text, image)")
    task_type: str = Field(
        ..., description="Type of ML task (e.g., classification, regression, clustering)"
    )
    language: Optional[str] = Field(
        default="python", description="Programming language for examples"
    )
    context: Optional[str] = Field(
        default=None, description="Additional context about the data task"
    )


class FeatureExample(BaseModel):
    """Model for a feature engineering example."""

    name: str = Field(..., description="Name of the feature engineering technique")
    description: str = Field(..., description="Description of the technique")
    code_example: str = Field(..., description="Example code implementing the technique")
    use_cases: List[str] = Field(..., description="Appropriate use cases for this technique")


class FeatureEngineeringResponse(BaseModel):
    """Response model for feature engineering guidance."""

    techniques: List[FeatureExample] = Field(
        ..., description="List of feature engineering techniques"
    )
    best_practices: List[str] = Field(..., description="Best practices for feature engineering")
    libraries: List[str] = Field(..., description="Recommended libraries to use")


@router.post("/techniques", response_model=FeatureEngineeringResponse)
async def get_feature_engineering_guidance(
    request: FeatureEngineeringRequest,
) -> Dict[str, Any]:
    """Get guidance on feature engineering techniques.

    Args:
        request: The feature engineering guidance request

    Returns:
        Dict containing techniques, best practices, and recommended libraries
    """
    # Example response for tabular data with classification task
    if request.data_type.lower() == "tabular" and request.task_type.lower() == "classification":
        return {
            "techniques": [
                {
                    "name": "One-hot encoding",
                    "description": "Convert categorical variables into binary vectors.",
                    "code_example": """
import polars as pl

# One-hot encoding with Polars
df = pl.read_csv("data.csv")

# Get the unique categories
categories = df["category_column"].unique().to_list()

# Create dummy columns
df_encoded = df.with_columns([
    pl.when(pl.col("category_column") == category)
    .then(1)
    .otherwise(0)
    .alias(f"category_{category}")
    for category in categories
])

# Alternative approach
df_encoded = df.to_dummies(columns=["category_column"])
""",
                    "use_cases": [
                        "Categorical features with no ordinal relationship",
                        "When the number of categories is manageable",
                    ],
                },
                {
                    "name": "Binning/Bucketing",
                    "description": "Convert continuous variables into discrete buckets.",
                    "code_example": """
import polars as pl
import numpy as np

# Load data
df = pl.read_csv("data.csv")

# Create bins for a numeric column - equal-width binning
bins = np.linspace(df["numeric_col"].min(), df["numeric_col"].max(), 5)
bin_labels = ["very_low", "low", "medium", "high"]

# Method 1: Using expressions
df = df.with_columns([
    pl.when(pl.col("numeric_col") < bins[1])
    .then(bin_labels[0])
    .when(pl.col("numeric_col") < bins[2])
    .then(bin_labels[1])
    .when(pl.col("numeric_col") < bins[3])
    .then(bin_labels[2])
    .otherwise(bin_labels[3])
    .alias("numeric_col_binned")
])

# Method 2: Using cut function
df = df.with_columns([
    pl.cut(
        pl.col("numeric_col"),
        breaks=bins,
        labels=bin_labels
    ).alias("numeric_col_binned")
])
""",
                    "use_cases": [
                        "When the relationship between feature and target is non-linear",
                        "To reduce the effect of outliers",
                        "To improve model interpretability",
                    ],
                },
                {
                    "name": "Feature scaling",
                    "description": "Standardize or normalize numeric features.",
                    "code_example": """
import polars as pl

# Load data
df = pl.read_csv("data.csv")

# Standardization (Z-score normalization)
numeric_cols = ["feature1", "feature2", "feature3"]
df = df.with_columns([
    ((pl.col(col) - pl.col(col).mean()) / pl.col(col).std()).alias(f"{col}_scaled")
    for col in numeric_cols
])

# Min-Max normalization
df = df.with_columns([
    ((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(f"{col}_normalized")
    for col in numeric_cols
])
""",
                    "use_cases": [
                        "When features have different scales",
                        "For algorithms sensitive to feature scales (e.g., SVM, k-means, neural networks)",
                    ],
                },
            ],
            "best_practices": [
                "Always split your data before performing feature engineering to prevent data leakage",
                "Create pipelines to ensure consistent feature engineering between training and inference",
                "Document all feature transformations for reproducibility",
                "Consider the impact of outliers when scaling features",
                "Select features based on their correlation with the target variable",
                "Use cross-validation to validate feature engineering effectiveness",
            ],
            "libraries": ["polars", "scikit-learn", "feature-engine"],
        }
    elif request.data_type.lower() == "text":
        return {
            "techniques": [
                {
                    "name": "TF-IDF Vectorization",
                    "description": "Convert text data into numerical features based on term frequency and inverse document frequency.",
                    "code_example": """
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl

# Load data
df = pl.read_csv("text_data.csv")
texts = df["text_column"].to_list()

# Create TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts)

# Convert to DataFrame
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pl.from_numpy(tfidf_matrix.toarray(), schema=feature_names)

# Combine with original DataFrame if needed
df_with_features = pl.concat([
    df, 
    tfidf_df
], how="horizontal")
""",
                    "use_cases": [
                        "Text classification",
                        "Document similarity",
                        "Information retrieval",
                    ],
                },
            ],
            "best_practices": [
                "Clean and normalize text data before feature extraction",
                "Consider removing stop words and applying stemming/lemmatization",
                "Use n-grams to capture phrase patterns",
                "Apply dimensionality reduction for large feature spaces",
            ],
            "libraries": ["scikit-learn", "spacy", "polars", "transformers"],
        }
    else:
        # Default generic response
        return {
            "techniques": [
                {
                    "name": "Feature Selection",
                    "description": "Select the most relevant features for your model.",
                    "code_example": """
from sklearn.feature_selection import SelectKBest, f_classif
import polars as pl
import numpy as np

# Load data
df = pl.read_csv("data.csv")
X = df.select(pl.exclude("target")).to_numpy()
y = df["target"].to_numpy()

# Select top k features
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_indices = selector.get_support(indices=True)
selected_features = df.columns[selected_indices]
print("Selected features:", selected_features)
""",
                    "use_cases": [
                        "High-dimensional datasets",
                        "Reducing overfitting",
                        "Improving model interpretability",
                    ],
                },
            ],
            "best_practices": [
                "Always validate feature engineering with cross-validation",
                "Consider both domain knowledge and data-driven approaches",
                "Document your feature engineering process",
            ],
            "libraries": ["polars", "scikit-learn", "numpy"],
        }
