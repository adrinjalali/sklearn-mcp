"""Router for general Python project guidelines for agents."""

import os


async def get_python_general_guidelines() -> str:
    """Get general Python guidelines for agent projects.

    Returns:
        str: Markdown content with general Python guidelines.
    """
    md_path = os.path.join(os.path.dirname(__file__), "python_general.md")
    with open(md_path, "r", encoding="utf-8") as f:
        markdown = f.read()
    return markdown
