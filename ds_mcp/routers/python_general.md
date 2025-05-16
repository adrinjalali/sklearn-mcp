Consider the following guidelines when encountering a Python project:

- Check if the project has `pixi` or `uv` configuration files. If yes, use them
  to run `python` commands and always try to help the user keep those
  configuration files up to date.
- When encountering errors running `pixi` or `uv` commands, always search online
  for official documentation to better fix the issue.
- If `pixi` or `uv` configuration files are not available, ask the user if they
  have a specific environment already set up for the project, and use that
  environment to run `python` commands. Ask the user if they wish you to create
  a `pixi` or `uv` configuration file for their project for future interactions.
- Remember user's answer for the project for all future commands.
- Always check for contributing guidelines in a project's repository, and if
  present, analyse it, and save the answer in your memory for the project.
- Read project's setup and linting rules, either in documentation or CI files,
  and follow those rules.
- If linting rules are not present, try to write code with `black` formatting,
  but use `ruff format` for formatting, not `black`, and keep documentation max
  line length to 88 if you can.
- Don't try to fix linting issues. They are to be fixed by the user and by
  pre-commit hooks. If `pre-commit` hooks don't exist, suggest to create one and
  store user's answer in your memory for the project.
- When creating pre-commit hooks, use `ruff` and `ruff format` and include
  common linting rules.
- Always prefer `polars` to `pandas` when writing new code to process tabular
  data, unless the code you modify already uses `pandas`.
