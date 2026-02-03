# Contributing to Advanced Fraud Detection System

Thank you for your interest in contributing! This guide helps you get started.

## Development Setup

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/advanced-fraud-detection-system.git
cd advanced-fraud-detection-system

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-cov black isort mypy

# Run tests
pytest tests/ -v

# Start development server
uvicorn app.main:app --reload
```

## Code Standards

### Style Guide

- **Formatter**: Black (line length 88)
- **Import Sorting**: isort
- **Type Hints**: Required for all functions
- **Docstrings**: Google style

```python
def detect_fraud(
    transaction: Transaction,
    request_id: str = ""
) -> FraudDetectionResult:
    """
    Detect fraud in a transaction.

    Args:
        transaction: Transaction to analyze
        request_id: Correlation ID for tracing

    Returns:
        FraudDetectionResult with decision and risk score

    Raises:
        FraudDetectionError: If detection fails
    """
```

### Architecture Rules

1. **Block Layer**: No I/O operations (database, HTTP, file system)
2. **Service Layer**: All I/O through dependency injection
3. **Utils Layer**: No business logic
4. **Error Handling**: Always use custom exceptions from `src/utils/error_handling.py`

### Testing Requirements

- Minimum 90% code coverage
- All new features require unit tests
- Integration tests for cross-component flows

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_device_fingerprint.py -v
```

## Branching Strategy

```
main (production)
├── develop (integration)
│   ├── feature/device-fingerprinting
│   ├── feature/payment-validation
│   └── bugfix/false-positive-rate
└── hotfix/critical-fix
```

### Branch Naming

- `feature/short-description` - New features
- `bugfix/issue-number` - Bug fixes
- `hotfix/critical-description` - Production hotfixes
- `refactor/component-name` - Code improvements

## Pull Request Process

1. Create a feature branch from `develop`
2. Make changes with tests
3. Ensure all tests pass: `pytest tests/ -v`
4. Run linting: `black . && isort .`
5. Create PR with description following template
6. Request review from 2+ team members
7. Address feedback
8. Merge after approval

### PR Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Coverage maintained at 90%+

## Checklist

- [ ] Code follows style guide
- [ ] Type hints added
- [ ] Docstrings updated
- [ ] CHANGELOG updated
```

## Commit Messages

Follow Conventional Commits:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Examples:

```
feat(fingerprint): add behavioral attribute collection
fix(payment): correct cross-border detection logic
docs(architecture): update deployment diagram
```

## Getting Help

- **Issues**: Create GitHub issue for bugs/features
- **Discussions**: Use GitHub Discussions for questions
- **Slack**: #fraud-detection-dev channel

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
