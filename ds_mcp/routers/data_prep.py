"""Router for data preparation endpoints."""

from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


class DataCleaningRequest(BaseModel):
    """Request model for data cleaning guidance."""

    data_type: str = Field(..., description="Type of data (e.g., tabular, text, image)")
    language: Optional[str] = Field(
        default="python", description="Programming language for examples"
    )
    context: Optional[str] = Field(
        default=None, description="Additional context about the data task"
    )


class BestPracticeItem(BaseModel):
    """Model for a single best practice item."""

    title: str = Field(..., description="Short title of the best practice")
    description: str = Field(..., description="Detailed description of the best practice")
    code_example: Optional[str] = Field(
        default=None, description="Example code demonstrating the best practice"
    )
    importance: str = Field(
        default="medium",
        description="Importance level of the practice (critical, high, medium, low)",
    )


class DataCleaningResponse(BaseModel):
    """Response model for data cleaning guidance."""

    best_practices: List[BestPracticeItem] = Field(
        ..., description="List of data cleaning best practices"
    )
    libraries: List[str] = Field(..., description="Recommended libraries to use")
    additional_resources: Optional[List[str]] = Field(
        default=None, description="Additional resources for learning"
    )


@router.post("/cleaning", response_model=DataCleaningResponse)
async def get_data_cleaning_guidance(request: DataCleaningRequest) -> Dict[str, Any]:
    """Get guidance on data cleaning best practices.

    Args:
        request: The data cleaning guidance request

    Returns:
        Dict containing best practices, recommended libraries, and resources
    """
    # Example response for tabular data
    if request.data_type.lower() == "tabular":
        return {
            "best_practices": [
                {
                    "title": "Use Polars for efficient data processing",
                    "description": "Polars is a fast DataFrame library that should be preferred over pandas for most data processing tasks. It provides better performance for large datasets.",
                    "code_example": """
import polars as pl

# Load data
df = pl.read_csv("data.csv")

# Basic cleaning
df_clean = df.select([
    pl.all().drop_nulls(),
    pl.all().filter(~pl.col("value").is_nan())
])

# Handle missing values
df_clean = df.select([
    pl.all(),
    pl.col("numeric_col").fill_null(pl.col("numeric_col").mean()),
    pl.col("categorical_col").fill_null("unknown")
])
""",
                    "importance": "critical",
                },
                {
                    "title": "Check for data types and convert if necessary",
                    "description": "Ensure all columns have the correct data types to prevent unexpected behavior in analysis.",
                    "code_example": """
import polars as pl

# Load data
df = pl.read_csv("data.csv")

# Check data types
print(df.schema)

# Convert data types
df = df.with_columns([
    pl.col("date_column").str.strptime(pl.Date, "%Y-%m-%d"),
    pl.col("numeric_as_string").cast(pl.Float64)
])
""",
                    "importance": "high",
                },
                {
                    "title": "Remove or handle duplicate rows",
                    "description": "Duplicate data can bias your analysis and models.",
                    "code_example": """
import polars as pl

# Load data
df = pl.read_csv("data.csv")

# Check for duplicates
n_duplicates = df.shape[0] - df.unique().shape[0]
print(f"Found {n_duplicates} duplicate rows")

# Remove duplicates
df_unique = df.unique()

# Or remove duplicates based on specific columns
df_unique_subset = df.unique(subset=["id", "timestamp"])
""",
                    "importance": "medium",
                },
            ],
            "libraries": ["polars", "pyarrow", "numpy"],
            "additional_resources": [
                "https://pola.rs/",
                "https://github.com/pola-rs/polars/tree/master/examples",
            ],
        }
    
    # Example response for text data
    elif request.data_type.lower() == "text":
        return {
            "best_practices": [
                {
                    "title": "Normalize text data",
                    "description": "Standardize case, remove extra whitespace, and perform other basic text normalization.",
                    "code_example": """
import re

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    # Remove special characters (optional, depends on use case)
    text = re.sub(r'[^\\w\\s]', '', text)
    return text

# Apply to dataset
df = df.with_column(pl.col("text").map_elements(normalize_text))
""",
                    "importance": "high",
                },
            ],
            "libraries": ["spacy", "nltk", "polars", "regex"],
            "additional_resources": [
                "https://spacy.io/usage/linguistic-features",
                "https://www.nltk.org/",
            ],
        }
    else:
        # Default generic response
        return {
            "best_practices": [
                {
                    "title": "Document your data cleaning steps",
                    "description": "Always document each transformation applied to your data for reproducibility.",
                    "code_example": None,
                    "importance": "high",
                }
            ],
            "libraries": ["polars", "numpy"],
            "additional_resources": [],
        }
