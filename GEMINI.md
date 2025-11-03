# Gemini Best Practices

This document outlines the best practices for contributing to the ERICA project, specifically when using Gemini for code generation and modification.

## 1. Code Style and Conventions

Adhere to the existing code style and conventions to maintain consistency.

*   **Typing:** Use type hints for all function signatures and variable declarations.
*   **Docstrings:** Write comprehensive docstrings for all public functions and classes using the Google docstring format.
*   **Naming:** Follow the existing naming conventions:
    *   `snake_case` for functions and variables.
    *   `PascalCase` for classes.
*   **Imports:** Organize imports into three sections: standard library, third-party libraries, and local application imports.
*   **Line Length:** Keep lines under 88 characters.

## 2. Project Structure

Maintain the existing project structure. When adding new functionality, consider whether it belongs in an existing module or a new one.

*   `erica/core.py`: Main `ERICA` class and orchestration logic.
*   `erica/clustering.py`: Clustering algorithms and related functions.
*   `erica/metrics.py`: Replicability metrics (CRI, WCRI, TWCRI).
*   `erica/data.py`: Data loading and preprocessing.
*   `erica/plotting.py`: Visualization functions.
*   `erica/utils.py`: Helper functions and utilities.

## 3. Modifying Existing Code

When modifying existing code, use the `replace` tool with sufficient context to ensure that the change is applied correctly.

*   **Context is Key:** Provide at least 3 lines of context before and after the code to be changed.
*   **Be Specific:** Create a detailed instruction that clearly explains the reason for the change.

## 4. Adding New Features

When adding a new feature, follow these steps:

1.  **Create a Plan:** Outline the changes you will make and the files you will modify.
2.  **Write the Code:** Implement the new feature, following the existing code style and conventions.
3.  **Add Tests:** Write unit tests for the new feature in the `tests/` directory.
4.  **Update Documentation:** If the new feature changes the public API, update the relevant docstrings and documentation.

## 5. Running Tests

After making any changes, run the existing tests to ensure that you have not introduced any regressions.

```bash
pytest
```

## 6. Example Gemini Prompts

### Modifying a Function

**Prompt:**

> In `erica/metrics.py`, I want to modify the `compute_cri` function. Instead of returning a `np.ndarray`, I want it to return a tuple of `(np.ndarray, float)` where the second element is the mean of the CRI values.

### Adding a New Function

**Prompt:**

> In `erica/plotting.py`, I want to add a new function called `plot_cri_distribution` that takes a `clam_matrix` and a `k` value and creates a box plot of the CRI values for each cluster.

By following these best practices, you can ensure that your contributions are consistent with the existing codebase and easy for other developers to understand and maintain.
