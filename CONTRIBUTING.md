# Contributing to Churn Prediction System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## 🎯 How to Contribute

### 1. Report a Bug
- **Go to:** [Issues](https://github.com/meghraj150198/Churn-Prediction-and-Retention-Strategy-for-a-Telecom-Provider/issues)
- **Click:** New Issue
- **Describe:**
  - What you were doing
  - What went wrong
  - Expected behavior
  - Screenshots (if applicable)
  - Your environment (Python version, OS)

### 2. Request a Feature
- **Go to:** [Issues](https://github.com/meghraj150198/Churn-Prediction-and-Retention-Strategy-for-a-Telecom-Provider/issues)
- **Click:** New Issue
- **Type:** Feature Request
- **Describe:**
  - What feature you'd like
  - Why it would be useful
  - How it should work
  - Any related issues

### 3. Submit Code Changes
1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Churn-Prediction-and-Retention-Strategy-for-a-Telecom-Provider.git
   cd Churn-Prediction-and-Retention-Strategy-for-a-Telecom-Provider
   ```

3. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make changes**
   - Follow PEP 8 style guide
   - Add comments for complex logic
   - Write docstrings for functions
   - Keep changes focused

5. **Test your changes**
   ```bash
   # Run any existing tests
   python -m pytest tests/

   # Manual testing
   python your_changes.py
   ```

6. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```
   
   Commit message format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation
   - `refactor:` Code refactoring
   - `test:` Adding tests
   - `chore:` Maintenance

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create Pull Request**
   - Go to original repository
   - Click "New Pull Request"
   - Compare your fork to main
   - Add description of changes
   - Submit PR

## 📋 Code Style Guidelines

### Python Style (PEP 8)
```python
# Good: Clear, documented, follows conventions
def calculate_churn_risk(customer_data: dict) -> float:
    """
    Calculate churn risk score for a customer.
    
    Args:
        customer_data: Dictionary with customer features
        
    Returns:
        float: Risk score between 0 and 1
    """
    risk_score = (
        customer_data['tenure_months'] * 0.1 +
        customer_data['complaints'] * 0.5
    )
    return min(risk_score, 1.0)


# Bad: Unclear, no documentation, inconsistent
def calc_r(d):
    r = d['t'] * 0.1 + d['c'] * 0.5
    return r
```

### Naming Conventions
```python
# Variables & Functions: lowercase with underscores
customer_id = "CUST_001"
def get_customer_data():
    pass

# Classes: PascalCase
class OperationalIntegration:
    pass

# Constants: UPPERCASE with underscores
MAX_RISK_SCORE = 100
DEFAULT_TIMEOUT = 30
```

### Comments & Docstrings
```python
# Bad: No documentation
def score_customer(data):
    x = data['a'] * 2 + data['b']
    return x

# Good: Clear documentation
def score_customer(customer_data: dict) -> float:
    """
    Score customer churn risk based on historical data.
    
    Args:
        customer_data: Contains 'attitude' and 'behavior' keys
        
    Returns:
        float: Risk score (0-100)
    """
    return customer_data['attitude'] * 2 + customer_data['behavior']
```

## 🧪 Testing

### Before Submitting PR
```bash
# 1. Test your changes work
python -c "import your_module; your_module.test_function()"

# 2. Check imports
python your_file.py

# 3. Verify no syntax errors
python -m py_compile your_file.py

# 4. Run existing tests
python -m pytest tests/
```

### Adding Tests
```python
# tests/test_your_feature.py
import unittest
from your_module import your_function

class TestYourFeature(unittest.TestCase):
    def test_basic_functionality(self):
        result = your_function(test_input)
        self.assertEqual(result, expected_output)
    
    def test_edge_cases(self):
        result = your_function(edge_case)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

## 📖 Documentation

### Update Documentation If:
- Adding new feature → Add to PUBLIC_GUIDE.md
- Changing API → Update OPERATIONAL_INTEGRATION_GUIDE.md
- Bug fix with workaround → Add to ISSUES_RESOLVED.md
- New model insights → Update MODEL_RECOMMENDATIONS.md

### Format:
```markdown
## Feature Name

**Purpose:** What it does

**Usage:**
\`\`\`python
# Code example
\`\`\`

**Parameters:**
- param1: Description
- param2: Description

**Returns:** What it returns
```

## 🔍 Review Checklist

Before submitting PR, ensure:
- ✅ Code follows PEP 8
- ✅ All changes tested
- ✅ No broken imports
- ✅ Documentation updated
- ✅ No hardcoded credentials
- ✅ No large files (>50MB)
- ✅ Commit messages are clear
- ✅ PR description is complete

## 💡 Development Tips

### Local Development Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Debugging
```python
# Use logging
import logging
logger = logging.getLogger(__name__)
logger.info("Debug message")
logger.error("Error message")

# Use IDE debugger
# Set breakpoints and debug in VS Code, PyCharm, etc.

# Print debugging (last resort)
print(f"DEBUG: value = {value}")
```

### Performance Tips
```python
# Bad: Inefficient
def process_data(data):
    result = []
    for item in data:
        if item not in result:  # O(n) check each time
            result.append(item)
    return result

# Good: Efficient
def process_data(data):
    return list(dict.fromkeys(data))  # O(n) total
```

## 🤝 Community Guidelines

### Be Respectful
- Be kind and respectful in discussions
- Assume good intent
- No harassment or discrimination
- Help others learn

### Stay Focused
- Keep discussions on-topic
- Search before asking duplicates
- Use threads to organize conversation
- Give credit to others' ideas

### Share Knowledge
- Help answer questions
- Review others' PRs
- Share useful resources
- Document solutions

## 📚 Additional Resources

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Markdown Syntax](https://www.markdownguide.org/)
- [Git Documentation](https://git-scm.com/doc)

## ❓ Questions?

- Check [Issues](https://github.com/meghraj150198/Churn-Prediction-and-Retention-Strategy-for-a-Telecom-Provider/issues)
- Read [PUBLIC_GUIDE.md](PUBLIC_GUIDE.md)
- Review [Documentation](.)
- Start a [Discussion](https://github.com/meghraj150198/Churn-Prediction-and-Retention-Strategy-for-a-Telecom-Provider/discussions)

## 🎉 Thank You!

Your contributions help make this project better for everyone. We appreciate your time and effort!

---

**Happy Contributing!** 🚀
