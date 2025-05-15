# Data Science Workflow Guidance

## Overview
A well-structured data science workflow consists of several key stages: data exploration and understanding, data preparation, feature engineering, model selection and training, evaluation, and deployment. Following best practices at each stage helps ensure reproducible, maintainable, and high-quality results.

## Workflow Stages

### Project Setup
**Description:** Setting up a well-organized project structure

**Best Practices:**
- Use a consistent project structure for all data science projects
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
- Explore data distributions, missingness, and relationships
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

<!-- Additional workflow stages, best practices, code examples, and pitfalls can be added here following the same structure. -->
