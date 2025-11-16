# Contributing to Quantum Trader Pro

## Code Quality Standards

### Style Guide

- **Python Version**: 3.9-3.11
- **Line Length**: 100 characters max
- **Formatter**: Black
- **Import Sorting**: isort (black profile)
- **Linter**: flake8

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Type Hints

All new code must include type hints. Use types from `utils/types.py`:

```python
from utils.types import Price, Size, OrderResult, Config

def calculate_position_size(
    capital: float,
    risk_percent: float,
    entry_price: Price,
    stop_loss: Price
) -> Size:
    """Calculate position size based on risk."""
    ...
```

### Error Handling

Use custom exceptions from `utils/exceptions.py`:

```python
from utils.exceptions import (
    InsufficientFundsError,
    RiskLimitExceededError,
    wrap_exception
)

def execute_trade(signal):
    try:
        # ... trading logic
    except Exception as e:
        raise wrap_exception(e, TradingError, {'signal': signal})
```

### Documentation

All public functions must have docstrings in Google style:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Short description of function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Example:
        >>> function_name(value1, value2)
        result
    """
```

### Testing

- All new features must include tests
- Run tests: `pytest tests/`
- Check coverage: `pytest --cov=. tests/`
- Minimum coverage: 80%

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Adding tests
- `docs`: Documentation
- `build`: Build system changes
- `ci`: CI/CD changes
- `perf`: Performance improvements
- `style`: Code style (formatting, etc.)

### Pull Request Process

1. Create feature branch from `main`
2. Implement changes with tests
3. Run `pre-commit run --all-files`
4. Run `pytest tests/`
5. Update documentation if needed
6. Create PR with clear description

### Security

- **NEVER** commit API keys or secrets
- Use `.env` for sensitive data
- Validate all user inputs
- Use `utils/security.py` for validation
- Run `bandit -r .` for security checks
