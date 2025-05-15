"""Router for general data science workflow guidance."""

from typing import Any
from typing import Dict


async def get_workflow_guidance(
    task_description: str, data_type: str = "tabular", context: str = None
) -> Dict[str, Any]:
    """Get guidance on data science workflow best practices.

    Args:
        task_description (str):
            A short natural language description of the data science or ML task (e.g.,
            'binary classification', 'time series forecasting', 'image segmentation',
            'recommendation system'). This should specify the main goal or problem type.
        data_type (str, optional):
            The type of data being used. Common values: 'tabular', 'text', 'image',
            'time series', etc. Defaults to 'tabular'.
        context (str, optional):
            Additional context, constraints, or details relevant to the task. This may
            include dataset characteristics, business requirements, performance
            constraints, or any other information that could influence workflow
            recommendations.

    Returns:
        dict: A dictionary containing workflow stages, best practices, recommended
        libraries, and common pitfalls for the specified task and data type.
    """
    # Example response for general data science workflow
    return {
        "overview": (
            "A well-structured data science workflow consists of several key stages: "
            "data exploration and understanding, data preparation, feature engineering, "
            "model selection and training, evaluation, and deployment. Following best "
            "practices at each stage helps ensure reproducible, maintainable, and "
            "high-quality results."
        ),
        "workflow_stages": [
            {
                "name": "Project Setup",
                "description": "Setting up a well-organized project structure",
                "best_practices": [
                    "Use a consistent project structure for all data science projects",
                    "Create a proper environment management setup (using pixi, uv, or micromamba)",
                    "Use version control (git) from the beginning",
                    "Document your project with a clear README",
                    "Set up a pre-commit configuration with ruff and ruff format for code quality",
                ],
                "code_example": "# Example project structure\nmy_project/\n├── README.md                  # Project documentation\n├── pixi.toml                  # Environment management\n├── .pre-commit-config.yaml    # Code quality checks\n├── data/                      # Data directory\n│   ├── raw/                   # Raw, immutable data\n│   └── processed/             # Cleaned and processed data\n├── notebooks/                 # Exploratory notebooks\n├── src/                       # Source code\n│   ├── __init__.py\n│   ├── data/                  # Data processing modules\n│   ├── features/              # Feature engineering\n│   ├── models/                # Model training and evaluation\n│   └── visualization/         # Visualization modules\n├── tests/                     # Tests for your code\n└── reports/                   # Generated analysis reports, figures",
                "common_pitfalls": [
                    "Starting without proper environment isolation",
                    "Not using version control from the beginning",
                    "Keeping data files in version control (use .gitignore for data)",
                    "Not documenting the project structure and setup steps",
                ],
            },
            {
                "name": "Data Exploration and Understanding",
                "description": "Exploring and understanding the data before processing",
                "best_practices": [
                    "Begin with questions, not techniques",
                    "Understand the data source, collection methods, and limitations",
                    "Explore data distributions, missingness, and relationships",
                    "Create visualizations to understand patterns and outliers",
                    "Document insights gained from exploration",
                    "Prefer polars over pandas for large datasets",
                ],
                "code_example": 'import polars as pl\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Load the data\ndf = pl.read_csv("data/raw/dataset.csv")\n\n# Basic information\nprint(f"Data shape: {df.shape}")\nprint(f"Data columns: {df.columns}")\nprint(f"Data types:\\n{df.dtypes}")\n\n# Summary statistics\nprint(df.describe())\n\n# Check for missing values\nmissing_values = df.null_count()\nprint(f"Missing values per column:\\n{missing_values}")\n\n# Visualize distributions for numerical columns\nnumeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns\nfor col in numeric_cols:\n    plt.figure(figsize=(10, 6))\n    sns.histplot(df[col].to_numpy(), kde=True)\n    plt.title(f\'Distribution of {col}\')\n    plt.xlabel(col)\n    plt.ylabel(\'Frequency\')\n    plt.tight_layout()\n    plt.show()',
                "common_pitfalls": [
                    "Rushing through exploration to get to modeling",
                    "Not checking for data quality issues early",
                    "Ignoring outliers without understanding them",
                    "Not examining correlations between features",
                    "Not visualizing the data",
                ],
            },
            {
                "name": "Data Preparation",
                "description": "Cleaning and preparing the data for analysis",
                "best_practices": [
                    "Handle missing values appropriately based on the context",
                    "Address outliers with justification",
                    "Convert categorical variables to appropriate formats",
                    "Scale numerical features when needed (e.g., for distance-based algorithms)",
                    "Split data into training, validation, and test sets before any transformations",
                    "Create reproducible preprocessing pipelines",
                    "Document all cleaning decisions and transformations",
                ],
                "code_example": 'import polars as pl\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.impute import SimpleImputer\n\n# Load the data\ndf = pl.read_csv("data/raw/dataset.csv")\n\n# Define features and target\nX = df.select(pl.exclude("target")).to_pandas()\ny = df["target"].to_pandas()\n\n# Split the data first to prevent data leakage\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\n# Identify column types\nnumeric_features = X.select_dtypes(include=[\'int64\', \'float64\']).columns\ncategorical_features = X.select_dtypes(include=[\'object\', \'category\']).columns\n\n# Create preprocessing pipelines\nnumeric_transformer = Pipeline(\n    steps=[\n        ("imputer", SimpleImputer(strategy="median")),\n        ("scaler", StandardScaler())\n    ]\n)\n\ncategorical_transformer = Pipeline(\n    steps=[\n        ("imputer", SimpleImputer(strategy="most_frequent")),\n        ("onehot", OneHotEncoder(handle_unknown="ignore"))\n    ]\n)\n\n# Combine preprocessing steps\npreprocessor = ColumnTransformer(\n    transformers=[\n        ("num", numeric_transformer, numeric_features),\n        ("cat", categorical_transformer, categorical_features)\n    ]\n)\n\n# Fit the preprocessing pipeline on training data only\nX_train_processed = preprocessor.fit_transform(X_train)\nX_test_processed = preprocessor.transform(X_test)',
                "common_pitfalls": [
                    "Data leakage from test to training set",
                    "Using inappropriate strategies for missing data",
                    "Not checking the distribution after transformations",
                    "Applying transformations inconsistently between training and test",
                    "Not saving preprocessing pipelines for reproducibility",
                ],
            },
            {
                "name": "Model Selection and Training",
                "description": "Selecting and training appropriate models",
                "best_practices": [
                    "Start with simple models before moving to complex ones",
                    "Use cross-validation to tune hyperparameters",
                    "Consider model interpretability requirements",
                    "Address class imbalance if present",
                    "Implement proper regularization to prevent overfitting",
                    "Track experiments with metrics and parameters",
                    "Create reproducible training workflows",
                ],
                "code_example": "import polars as pl\nfrom sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score, GridSearchCV\nfrom sklearn.metrics import classification_report, roc_auc_score\n\n# Start with a simple baseline model\nbaseline = LogisticRegression(max_iter=1000, random_state=42)\nbaseline_scores = cross_val_score(baseline, X_train_selected, y_train, cv=5, scoring='roc_auc')\nprint(f\"Baseline ROC AUC: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}\")\n\n# Try more complex models\nmodels = {\n    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n    'Random Forest': RandomForestClassifier(random_state=42),\n    'Gradient Boosting': GradientBoostingClassifier(random_state=42)\n}\n\n# Compare models with cross-validation\nfor name, model in models.items():\n    scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')\n    print(f\"{name} ROC AUC: {scores.mean():.4f} ± {scores.std():.4f}\")\n\n# Hyperparameter tuning for the best model\nbest_model = GradientBoostingClassifier(random_state=42)  # Assume this was the best\nparam_grid = {\n    'n_estimators': [100, 200, 300],\n    'learning_rate': [0.01, 0.05, 0.1],\n    'max_depth': [3, 5, 7],\n    'min_samples_split': [2, 5, 10]\n}\n\ngrid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)\ngrid_search.fit(X_train_selected, y_train)\nprint(f\"Best parameters: {grid_search.best_params_}\")\nprint(f\"Best cross-validation score: {grid_search.best_score_:.4f}\")",
                "common_pitfalls": [
                    "Using complex models without trying simpler baselines first",
                    "Incorrect cross-validation setup leading to data leakage",
                    "Overfitting to the validation set through excessive tuning",
                    "Not considering the computational requirements of models",
                    "Ignoring model interpretability requirements",
                    "Failing to save models and parameters for reproducibility",
                ],
            },
            {
                "name": "Model Evaluation",
                "description": "Thoroughly evaluating model performance",
                "best_practices": [
                    "Use multiple metrics appropriate for the problem",
                    "Analyze errors and edge cases",
                    "Consider business impact, not just statistical performance",
                    "Evaluate model fairness across subgroups if applicable",
                    "Perform sensitivity analysis for important features",
                    "Create clear visualizations of model performance",
                ],
                "code_example": "import matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score\n\n# Detailed evaluation of the model\ny_pred = best_model.predict(X_test_selected)\ny_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]\n\n# Confusion matrix with visualization\ncm = confusion_matrix(y_test, y_pred)\nplt.figure(figsize=(8, 6))\nsns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)\nplt.xlabel('Predicted labels')\nplt.ylabel('True labels')\nplt.title('Confusion Matrix')\nplt.savefig('reports/confusion_matrix.png')\nplt.show()\n\n# ROC Curve\nfpr, tpr, _ = roc_curve(y_test, y_pred_proba)\nroc_auc = roc_auc_score(y_test, y_pred_proba)\n\nplt.figure(figsize=(8, 6))\nplt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')\nplt.plot([0, 1], [0, 1], 'k--')\nplt.xlim([0.0, 1.0])\nplt.ylim([0.0, 1.05])\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('Receiver Operating Characteristic (ROC) Curve')\nplt.legend(loc=\"lower right\")\nplt.savefig('reports/roc_curve.png')\nplt.show()",
                "common_pitfalls": [
                    "Relying on a single metric for evaluation",
                    "Not considering the business context of errors",
                    "Ignoring model confidence in predictions",
                    "Not investigating specific failure cases",
                    "Focusing only on aggregate metrics rather than subgroup performance",
                ],
            },
            {
                "name": "Model Documentation and Deployment",
                "description": "Documenting and preparing the model for deployment",
                "best_practices": [
                    "Create comprehensive documentation of the model and process",
                    "Document model limitations and assumptions",
                    "Save all artifacts (preprocessors, models, evaluation results)",
                    "Implement version control for models",
                    "Create reproducible inference pipelines",
                    "Plan for model monitoring and retraining",
                ],
                "code_example": 'import json\nimport datetime\nimport os\n\n# Create a model card/documentation\nmodel_documentation = {\n    "model_name": "Depression Status Predictor",\n    "version": "1.0.0",\n    "creation_date": datetime.datetime.now().isoformat(),\n    "author": "Data Science Team",\n    "description": "Classification model to predict depression status",\n    "model_type": "Gradient Boosting Classifier",\n    "features": list(selected_features),\n    "target": "Depression_Status",\n    "metrics": {\n        "roc_auc": float(roc_auc),\n        "average_precision": float(average_precision)\n    },\n    "preprocessing": {\n        "missing_values": "Imputed using median for numeric, most frequent for categorical",\n        "scaling": "StandardScaler applied to numeric features",\n        "categorical_encoding": "OneHotEncoder for categorical features",\n        "feature_selection": "SelectKBest with f_classif, 20 features selected"\n    },\n    "limitations": [\n        "Model has not been tested on students from different cultural backgrounds",\n        "Predictions should be used as supportive information only, not as definitive diagnosis"\n    ],\n    "intended_use": "For educational and research purposes only"\n}\n\n# Save the model documentation\nos.makedirs("models", exist_ok=True)\nwith open("models/model_card.json", "w") as f:\n    json.dump(model_documentation, f, indent=4)',
                "common_pitfalls": [
                    "Insufficient documentation of model limitations",
                    "Not creating a complete inference pipeline",
                    "Not versioning model artifacts",
                    "Overlooking model interpretability for stakeholders",
                    "No plan for model monitoring and updating",
                ],
            },
        ],
        "libraries": [
            "polars",
            "scikit-learn",
            "skrub",
            "skops",
            "matplotlib",
            "seaborn",
            "numpy",
            "joblib",
            "optuna",
            "shap",
        ],
        "project_structure": {
            "data/": "Raw and processed data files",
            "notebooks/": "Exploratory Jupyter notebooks",
            "src/": "Source code for the project",
            "models/": "Saved model artifacts and documentation",
            "reports/": "Generated reports and visualizations",
            "tests/": "Tests for the codebase",
        },
        "additional_resources": [
            "https://skrub-data.org/stable/",
            "https://skops.readthedocs.io/en/latest/",
            "https://scikit-learn.org/stable/index.html",
            "https://pola.rs/",
            "https://pandas.pydata.org/docs/user_guide/index.html",
            "https://matplotlib.org/stable/tutorials/index.html",
            "https://seaborn.pydata.org/tutorial.html",
        ],
    }
