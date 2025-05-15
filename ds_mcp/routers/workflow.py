"""Router for general data science workflow guidance."""


import os


async def get_workflow_guidance(
    task_description: str, data_type: str = "tabular", context: str = None
) -> str:
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
        str: Markdown content with workflow guidance.
    """
    # Read the markdown file with workflow guidance
    md_path = os.path.join(os.path.dirname(__file__), "workflow_guidance.md")
    with open(md_path, "r", encoding="utf-8") as f:
        markdown = f.read()
    return markdown
