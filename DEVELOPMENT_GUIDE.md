# Development and Deployment Guide
## Soccer Prediction System - Developer Documentation

**Version:** Production v1.0  
**Status:** ✅ OPERATIONAL  
**Target Audience:** Developers, DevOps, System Administrators

---

## 🚀 Quick Setup

### Local Development Environment
```bash
# Clone repository
git clone <repository-url>
cd Soccer

# Install dependencies
pip install -r requirements.txt

# Verify installation
python final_system_validation.py
```

### Requirements
```txt
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0
```

---

## 🏗️ Development Workflow

### Setting Up Development Environment

#### 1. Python Environment
```bash
# Create virtual environment
python -m venv soccer_env
source soccer_env/bin/activate  # Linux/Mac
# or
soccer_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. IDE Configuration
**VS Code Settings:**
```json
{
    "python.defaultInterpreterPath": "./soccer_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

#### 3. Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### Development Standards

#### Code Style
- **PEP 8 Compliance:** Use `black` formatter
- **Type Hints:** All functions must have type annotations
- **Docstrings:** Google-style docstrings required
- **Logging:** Use structured logging with appropriate levels

#### Example Code Template
```python
"""Module docstring describing purpose."""

import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class ExampleClass:
    """Class docstring describing purpose and usage."""
    
    def __init__(self, param: str) -> None:
        """Initialize with parameter.
        
        Args:
            param: Description of parameter
        """
        self.param = param
    
    def example_method(self, data: Dict[str, Union[str, float]]) -> Optional[float]:
        """Example method with full type annotations.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Processed result or None if error
            
        Raises:
            ValueError: If data is invalid
        """
        try:
            result = self._process_data(data)
            logger.info(f"Processed data successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
```

---

## 🧪 Testing Framework

### Test Structure
```
tests/
├── unit/
│   ├── test_elo_system.py
│   ├── test_model_loading.py
│   └── test_feature_engineering.py
├── integration/
│   ├── test_end_to_end.py
│   └── test_api_endpoints.py
└── performance/
    ├── test_memory_usage.py
    └── test_prediction_speed.py
```

### Running Tests

#### Full Test Suite
```bash
# Run all tests
python final_system_validation.py

# Run with verbose output
python final_system_validation.py --verbose

# Run specific test category
python comprehensive_corner_test.py
```

#### Individual Components
```bash
# Test ELO system
python debug_elo_keys.py

# Test model loading
python test_model_loading.py

# Test basic functionality
python simple_corner_test.py
```

### Test Development Guidelines

#### Unit Test Template
```python
import unittest
from unittest.mock import Mock, patch
from soccer_prediction.voting_ensemble_corners import VotingEnsembleCorners

class TestVotingEnsembleCorners(unittest.TestCase):
    """Test cases for VotingEnsembleCorners class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = VotingEnsembleCorners()
        self.sample_match = {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea'
        }
    
    def test_predict_corners_valid_input(self):
        """Test corner prediction with valid input."""
        result = self.predictor.predict_corners(self.sample_match)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
        self.assertLess(result, 20)
    
    @patch('soccer_prediction.auto_updating_elo.get_elo_data_with_auto_rating')
    def test_elo_integration(self, mock_elo):
        """Test ELO system integration."""
        mock_elo.return_value = {
            'home_elo': 1500,
            'away_elo': 1450,
            'elo_expected_goal_diff': 0.1
        }
        result = self.predictor.predict_corners(self.sample_match)
        self.assertIsNotNone(result)
```

---

## 📦 Deployment

### Production Deployment

#### Docker Configuration
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create models directory
RUN mkdir -p models

# Set permissions
RUN chmod +x *.py

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from voting_ensemble_corners import VotingEnsembleCorners; VotingEnsembleCorners()"

EXPOSE 8000

CMD ["python", "app.py"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  soccer-prediction:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - MODEL_DIR=/app/models
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - soccer-prediction
```

### API Service Implementation

#### Flask API
```python
from flask import Flask, request, jsonify, Response
from voting_ensemble_corners import VotingEnsembleCorners
from auto_updating_elo import get_elo_data_with_auto_rating
import logging
import json
import time

app = Flask(__name__)
predictor = VotingEnsembleCorners()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Quick system validation
        test_match = {'home_team': 'Test', 'away_team': 'Test'}
        predictor._prepare_features(test_match)
        return jsonify({'status': 'healthy', 'timestamp': time.time()})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/predict/corners', methods=['POST'])
def predict_corners():
    """Corner prediction endpoint."""
    try:
        data = request.json
        
        # Validate input
        required_fields = ['home_team', 'away_team']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Make prediction
        prediction = predictor.predict_corners(data)
        
        return jsonify({
            'corners': prediction,
            'home_team': data['home_team'],
            'away_team': data['away_team'],
            'timestamp': time.time()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/elo', methods=['POST'])
def get_elo():
    """ELO ratings endpoint."""
    try:
        data = request.json
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Missing home_team or away_team'}), 400
        
        elo_data = get_elo_data_with_auto_rating(home_team, away_team)
        return jsonify(elo_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
```

### Environment Configuration

#### Production Environment Variables
```bash
# Application settings
FLASK_ENV=production
LOG_LEVEL=INFO
MODEL_DIR=/app/models

# Performance settings
WORKERS=4
TIMEOUT=30
MAX_REQUESTS=1000

# Security settings
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com
```

#### Configuration Management
```python
import os
from typing import Optional

class Config:
    """Application configuration."""
    
    # Flask settings
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-key')
    DEBUG: bool = os.getenv('FLASK_ENV') != 'production'
    
    # Model settings
    MODEL_DIR: str = os.getenv('MODEL_DIR', 'models')
    
    # Logging settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    
    # Performance settings
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '60'))
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
```

---

## 📊 Monitoring and Logging

### Logging Configuration
```python
import logging
import sys
from datetime import datetime

def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """Set up application logging."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler (if specified)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers
    )
```

### Performance Monitoring
```python
import time
import psutil
import functools
from typing import Callable, Any

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            duration = end_time - start_time
            memory_diff = end_memory - start_memory
            
            logging.info(f"Performance: {func.__name__} - "
                        f"Duration: {duration:.3f}s, "
                        f"Memory: {memory_diff/1024/1024:.1f}MB, "
                        f"Success: {success}")
        
        return result
    return wrapper
```

### Health Monitoring Script
```python
#!/usr/bin/env python3
"""System health monitoring script."""

import requests
import time
import logging
from typing import Dict, Any

class HealthMonitor:
    """Monitor system health and performance."""
    
    def __init__(self, base_url: str = 'http://localhost:8000'):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
    
    def check_health(self) -> Dict[str, Any]:
        """Check system health."""
        try:
            response = requests.get(f'{self.base_url}/health', timeout=5)
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    def test_prediction(self) -> Dict[str, Any]:
        """Test prediction endpoint."""
        try:
            test_data = {
                'home_team': 'Arsenal',
                'away_team': 'Chelsea'
            }
            
            start_time = time.time()
            response = requests.post(
                f'{self.base_url}/predict/corners',
                json=test_data,
                timeout=10
            )
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'success',
                    'prediction': data.get('corners'),
                    'response_time': duration
                }
            else:
                return {
                    'status': 'error',
                    'status_code': response.status_code,
                    'response_time': duration
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

if __name__ == '__main__':
    monitor = HealthMonitor()
    
    # Check health
    health = monitor.check_health()
    print(f"Health: {health}")
    
    # Test prediction
    prediction = monitor.test_prediction()
    print(f"Prediction: {prediction}")
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: Soccer Prediction CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ --cov=./ --cov-report=xml
    
    - name: Run system validation
      run: |
        python final_system_validation.py

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t soccer-prediction:latest .
    
    - name: Deploy to production
      run: |
        # Add deployment steps here
        echo "Deploying to production..."
```

### Deployment Scripts

#### Deploy Script
```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

echo "Starting deployment..."

# Build Docker image
docker build -t soccer-prediction:latest .

# Stop existing container
docker stop soccer-prediction || true
docker rm soccer-prediction || true

# Run new container
docker run -d \
  --name soccer-prediction \
  -p 8000:8000 \
  -v ./models:/app/models:ro \
  -v ./logs:/app/logs \
  --restart unless-stopped \
  soccer-prediction:latest

# Wait for health check
sleep 10

# Verify deployment
curl -f http://localhost:8000/health || exit 1

echo "Deployment completed successfully!"
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```bash
# Check model files
ls -la models/
# Expected: random_forest_corners.pkl, xgboost_corners.pkl

# Test model loading
python test_model_loading.py

# Check permissions
chmod 644 models/*.pkl
```

#### 2. ELO System Errors
```bash
# Test ELO system
python debug_elo_keys.py

# Check for correct key names
# Should see: elo_win_probability, elo_expected_goal_diff, etc.
```

#### 3. Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Optimize if needed
# Consider model caching strategies
```

#### 4. Performance Problems
```bash
# Profile prediction time
python -c "
import time
from voting_ensemble_corners import VotingEnsembleCorners

predictor = VotingEnsembleCorners()
start = time.time()
result = predictor.predict_corners({'home_team': 'Arsenal', 'away_team': 'Chelsea'})
print(f'Prediction time: {time.time() - start:.3f}s')
"
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
python final_system_validation.py --debug
```

---

## 📈 Performance Optimization

### Memory Optimization
- **Model Caching:** Load models once, reuse for multiple predictions
- **Feature Engineering:** Optimize data structures and algorithms
- **Garbage Collection:** Explicit cleanup of large objects

### Speed Optimization
- **Vectorization:** Use NumPy operations where possible
- **Caching:** Cache frequently accessed data
- **Lazy Loading:** Load components only when needed

### Scalability Considerations
- **Stateless Design:** No session state between requests
- **Horizontal Scaling:** Multiple worker processes
- **Load Balancing:** Distribute requests across instances

---

This development guide provides everything needed to set up, develop, test, and deploy the soccer prediction system in a production environment.
