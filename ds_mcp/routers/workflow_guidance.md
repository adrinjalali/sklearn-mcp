# Data Science Workflow Guidance

## Overview
A well-structured data science workflow consists of several key stages: data exploration and understanding, data preparation, feature engineering, model selection and training, evaluation, and deployment. Following best practices at each stage helps ensure reproducible, maintainable, and high-quality results.

## Recommended Libraries & Documentation

- [skrub](https://skrub-data.org/stable/) — Machine learning for dataframes
- [skore](https://docs.skore.probabl.ai/0.8/) — Diagnostics, reporting, and best practices for scikit-learn workflows
- [skops](https://skops.readthedocs.io/en/stable/) — Secure model persistence and sharing
- [scikit-learn](https://scikit-learn.org/stable/) — Machine learning in Python
- [polars](https://pola.rs/) — Fast DataFrames for Python
- [ruff](https://docs.astral.sh/ruff/) — Fast Python linter and formatter
- [pixi](https://prefix.dev/docs/pixi/) — Fast, reproducible Python environments
- [uv](https://uv.pypa.io/en/stable/) — Ultra fast Python package installer and resolver
- [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) — Fast conda-compatible package manager

**Tip:** When using any recommended library, always check the official documentation and online resources for the latest best practices, usage patterns, and code examples.


---

## Estimator Selection Best Practices

- **Prefer** `sklearn.ensemble.HistGradientBoostingClassifier` and `sklearn.ensemble.HistGradientBoostingRegressor` over `GradientBoostingClassifier`, `GradientBoostingRegressor`, and all RandomForest classes (`RandomForestClassifier`, `RandomForestRegressor`).
- For linear models, **prefer** `sklearn.linear_model.SGDClassifier` and `sklearn.linear_model.SGDRegressor` over their non-SGD counterparts (e.g., `LogisticRegression`, `LinearRegression`, etc.).

These choices generally offer better scalability, speed, and support for large datasets. See the [scikit-learn documentation](https://scikit-learn.org/stable/) for details.

---

## Workflow Stages

### Project Setup
**Description:** Setting up a well-organized project structure

**Best Practices:**
- Use a consistent project structure for all data science projects
- Prefer `polars` over `pandas` for data processing, as it can be much faster and more efficient for large datasets
- Create a proper environment management setup (using pixi, uv, or micromamba)
- Use version control (git) from the beginning
- Document your project with a clear README
- Set up a pre-commit configuration with ruff and ruff format for code quality

**Code Example:**
```text
# Example project structure
my_project/
├── README.md                  # Project documentation
├── pixi.toml                  # Environment management
├── .pre-commit-config.yaml    # Code quality checks
├── data/                      # Data directory
│   ├── raw/                   # Raw, immutable data
│   └── processed/             # Cleaned and processed data
├── src/                       # Source code
│   ├── __init__.py
│   ├── data/                  # Data processing modules
│   ├── features/              # Feature engineering
│   ├── models/                # Model training and evaluation
│   └── visualization/         # Visualization modules
├── tests/                     # Tests for your code
└── reports/                   # Generated analysis reports, figures
```

**Common Pitfalls:**
- Starting without proper environment isolation
- Not using version control from the beginning
- Keeping data files in version control (use .gitignore for data)
- Not documenting the project structure and setup steps

### Data Exploration and Understanding
**Description:** Exploring and understanding the data before processing

**Best Practices:**
- Begin with questions, not techniques
- Understand the data source, collection methods, and limitations
- Explore data distributions, missing values, and relationships
- Create visualizations to understand patterns and outliers
- Document insights gained from exploration
- Prefer polars over pandas for large datasets

**Code Example:**
```python
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pl.read_csv("data/raw/dataset.csv")

# Basic information
print(f"Data shape: {df.shape}")
print(f"Data columns: {df.columns}")
print(f"Data types:\n{df.dtypes}")

# Summary statistics
print(df.describe())

# Check for missing values
missing_values = df.null_count()
print(f"Missing values per column:\n{missing_values}")

# Visualize distributions for numerical columns
numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col].to_numpy(), kde=True)
```


### Best Practices: Using skrub for Data Preparation and Feature Engineering
**Description:** skrub simplifies preprocessing and feature engineering for heterogeneous tabular data.

**Best Practices:**
- Use `TableVectorizer` for automatic, robust preprocessing of mixed-type data (numeric, categorical, text, dates).
- Prefer `GapEncoder` or `MinHashEncoder` for high-cardinality or messy categorical/text features.
- Use `DatetimeEncoder` to extract features from datetime columns.
- Integrate skrub transformers into scikit-learn pipelines for reproducibility and efficiency.
- Use `TableReport` for interactive data exploration and understanding your dataframe.
- For joining tables with messy keys, use skrub's `Joiner` or `fuzzy_join` for fuzzy matching.
- Always fit transformers on training data only to prevent data leakage.

**Code Example:**
```python
import polars as pl
from skrub import TableVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Load your data
# TableVectorizer expects pandas DataFrames
pl_df = pl.read_csv("data/raw/dataset.csv")
df = pl_df.to_pandas()

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Create a pipeline with TableVectorizer and a regressor
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=42)
)

# Fit and evaluate
pipeline.fit(X_train, y_train)
print("Test score:", pipeline.score(X_test, y_test))
```


### Reproducibility: Pipelines First
**Description:** Avoid performing data manipulations on the original dataframe before passing it to a scikit-learn pipeline. All possible data preparation and feature engineering steps should be encapsulated within a single pipeline. This ensures reproducibility, auditability, and prevents subtle data leakage or inconsistencies between training and inference.

**Best Practices:**
- Do not modify your dataframe (e.g., imputation, encoding, scaling) before passing it to a pipeline.
- Use scikit-learn (or compatible) transformers for all preprocessing steps.
- Compose all steps into a single `Pipeline` or `ColumnTransformer`.
- Save and reuse the pipeline for inference and deployment.

**Anti-pattern (to avoid):**
```python
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

# BAD: Data is modified outside the pipeline
train_df = pl.read_csv("data/train.csv")
train_df = train_df.fill_null(0)  # Imputation outside pipeline
X = train_df.drop("target").to_numpy()
y = train_df["target"].to_numpy()

pipe = Pipeline([
    ("clf", HistGradientBoostingClassifier())
])
pipe.fit(X, y)
```

**Correct approach:**
```python
import polars as pl
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

train_df = pl.read_csv("data/train.csv")
X = train_df.drop("target").to_numpy()
y = train_df["target"].to_numpy()

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ("clf", HistGradientBoostingClassifier())
])
pipe.fit(X, y)
```

All preprocessing (imputation) is now inside the pipeline, ensuring reproducibility and preventing leakage.


### Best Practices for Model Persistence with skops.io
**Description:** Persist and share models securely using the skops format.

**Best Practices:**
- Use `skops.io.dump` to persist trained models instead of pickle/joblib, especially when sharing or deploying.
- The `.skops` format is secure and recommended for reproducible, shareable machine learning workflows.
- Always document the model version and relevant metadata alongside the persisted file.

**Code Example:**
```python
import skops.io as sio

# Save your trained model (e.g., pipeline or estimator)
sio.dump(model, "model.skops")

# Load the model later
loaded_model = sio.load("model.skops")
```


### Best Practices: Model Evaluation and Diagnostics with skore
**Description:** Use skore to enhance your scikit-learn workflow with robust evaluation, diagnostics, and reporting tools.

**Best Practices:**
- Use `skore.EstimatorReport` to generate detailed reports on your estimator's performance and avoid common pitfalls.
- Use `skore.CrossValidationReport` for insightful diagnostics on cross-validation results.
- Use `skore.ComparisonReport` to benchmark and compare multiple estimator reports on the same test set.
- Use `skore.train_test_split` to get additional diagnostics when splitting your data.
- Track your machine learning results and experiments using `skore.Project` for reproducibility and organization.

**Code Example:**
```python
from skore import EstimatorReport, CrossValidationReport, ComparisonReport, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris

# Load data and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train and evaluate with EstimatorReport
rf = RandomForestClassifier(random_state=0)
rf_report = EstimatorReport(rf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
rf_report.display()  # Interactive report in notebook or HTML

# Cross-validation diagnostics
cv_report = CrossValidationReport(rf, X=X, y=y)
cv_report.display()

# Compare with another estimator
gb = GradientBoostingClassifier(random_state=0)
gb_report = EstimatorReport(gb, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
comparator = ComparisonReport(reports=[rf_report, gb_report])
comparator.display()
```

For more details and advanced usage, see the [skore documentation](https://docs.skore.probabl.ai/0.8/).

<!-- Additional workflow stages, best practices, code examples, and pitfalls can be added here following the same structure. -->
