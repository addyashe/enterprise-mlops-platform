"""Contributing to Enterprise MLOps Platform."""

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/my-feature`
5. Submit a pull request

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Code Standards

### Style Guide
- Follow PEP 8
- Use type hints for all functions
- Max line length: 100 characters
- Document all public functions with docstrings

### Linting
```bash
flake8 src tests
black src tests
mypy src --ignore-missing-imports
```

### Testing
- Write tests for all new code
- Minimum 80% code coverage
- Use pytest for test framework

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Pull Request Process

1. Update README.md with any new features
2. Update documentation as needed
3. Ensure all tests pass
4. Code review from at least one maintainer
5. Merge after approval

## Reporting Bugs

- Use GitHub Issues to report bugs
- Include:
  - Python version
  - OS/Environment
  - Steps to reproduce
  - Expected vs actual behavior
  - Error traceback

## Feature Requests

- Use GitHub Discussions for feature ideas
- Describe the use case and motivation
- Propose implementation approach if possible

## Code Review Checklist

- [ ] Tests pass
- [ ] Code follows style guide
- [ ] Docstrings present
- [ ] Type hints included
- [ ] No unnecessary complexity
- [ ] Error handling appropriate
