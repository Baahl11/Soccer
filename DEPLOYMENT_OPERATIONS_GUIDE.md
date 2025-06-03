# Deployment and Operations Guide

## Soccer Prediction Platform - FastAPI Backend
**Production Deployment Guide**  
**Version:** 1.0.0  
**Date:** June 2, 2025  

---

## Prerequisites

### System Requirements
- Python 3.8+ (Tested with Python 3.12.5)
- PostgreSQL 12+ database
- Redis server (for caching and Celery)
- 4GB+ RAM recommended
- 10GB+ disk space

### Required Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@host:5432/dbname
POSTGRES_USER=your_database_user
POSTGRES_PASSWORD=your_secure_password
POSTGRES_HOST=your_database_host
POSTGRES_PORT=5432
POSTGRES_DB=soccer_predictions

# Redis Configuration
REDIS_URL=redis://your_redis_host:6379/0
CELERY_BROKER_URL=redis://your_redis_host:6379/1
CELERY_RESULT_BACKEND=redis://your_redis_host:6379/2

# Security
SECRET_KEY=your_very_secure_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Stripe Payment Configuration
STRIPE_SECRET_KEY=sk_live_your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret
STRIPE_PUBLIC_KEY=pk_live_your_public_key

# External APIs
FOOTBALL_API_KEY=your_football_api_key
FOOTBALL_API_BASE_URL=https://api-football-v1.p.rapidapi.com/v3

# Email Configuration (Optional)
SMTP_HOST=your_smtp_host
SMTP_PORT=587
SMTP_USER=your_email_user
SMTP_PASSWORD=your_email_password

# Frontend Configuration
FRONTEND_URL=https://your-frontend-domain.com
ALLOWED_ORIGINS=["https://your-frontend-domain.com","https://www.your-domain.com"]
```

---

## Installation Steps

### 1. Clone and Setup
```bash
# Navigate to project directory
cd c:\Users\gm_me\Soccer\fastapi_backend

# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install additional packages (if not in requirements.txt)
pip install "pydantic[email]"
pip install stripe
```

### 2. Environment Configuration
```bash
# Create .env file in project root
cp .env.example .env

# Edit .env with your configuration values
# Use secure values for production!
```

### 3. Database Setup
```bash
# Run database migrations (if using Alembic)
alembic upgrade head

# Or create tables directly
python -c "
from app.core.database import Base, engine
Base.metadata.create_all(bind=engine)
print('Database tables created successfully')
"
```

### 4. Verification
```bash
# Test all imports and configuration
python -c "
from app.main import app
from app.core.config import get_settings
settings = get_settings()
print(f'✅ Backend ready - MODEL_VERSION: {settings.MODEL_VERSION}')
"
```

---

## Production Deployment

### Option 1: Docker Deployment (Recommended)

#### Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/soccer_predictions
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs

  db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
      POSTGRES_DB: soccer_predictions
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Option 2: Direct Server Deployment

#### Using Gunicorn (Production WSGI Server)
```bash
# Install Gunicorn
pip install gunicorn

# Start with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# Or with more configuration
gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile - \
  --log-level info
```

#### Using Systemd Service
```ini
# /etc/systemd/system/soccer-api.service
[Unit]
Description=Soccer Prediction API
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/path/to/fastapi_backend
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 127.0.0.1:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Option 3: Cloud Platform Deployment

#### Heroku
```bash
# Create Procfile
echo "web: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:\$PORT" > Procfile

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

#### AWS/GCP/Azure
- Use container registry for Docker images
- Configure load balancer and auto-scaling
- Set up managed database services
- Configure environment variables in cloud console

---

## Monitoring and Logging

### Application Monitoring
```python
# Add to main.py for health checks
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.MODEL_VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics():
    # Add application metrics
    return {
        "active_users": await get_active_user_count(),
        "predictions_today": await get_daily_prediction_count(),
        "uptime": get_uptime()
    }
```

### Logging Configuration
```python
# Enhanced logging setup
import logging
from logging.handlers import RotatingFileHandler

# Configure rotating file handler
handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)

# Set logging format
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s: %(message)s'
)
handler.setFormatter(formatter)

# Add to root logger
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)
```

### Performance Monitoring
```bash
# Install monitoring tools
pip install prometheus_client
pip install sentry-sdk[fastapi]

# Add to requirements.txt
```

---

## Database Management

### Backup Strategy
```bash
# Daily database backup
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB > $BACKUP_DIR/backup_$DATE.sql
gzip $BACKUP_DIR/backup_$DATE.sql
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

### Migration Management
```bash
# Using Alembic for schema migrations
alembic init alembic
alembic revision --autogenerate -m "Initial migration"
alembic upgrade head

# Check current migration status
alembic current
alembic history
```

---

## Security Considerations

### SSL/TLS Configuration
```nginx
# Nginx configuration for HTTPS
server {
    listen 443 ssl http2;
    server_name your-api-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Security Headers
```python
# Add security middleware to main.py
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)
```

### Rate Limiting
```python
# Install slowapi for rate limiting
pip install slowapi

# Add to main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply to endpoints
@app.get("/api/predictions")
@limiter.limit("100/minute")
async def get_predictions(request: Request):
    pass
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors
```bash
# Symptoms: ModuleNotFoundError when starting
# Solution: Verify PYTHONPATH and working directory
export PYTHONPATH="${PYTHONPATH}:/path/to/fastapi_backend"
cd /path/to/fastapi_backend
```

#### Issue 2: Database Connection Errors
```bash
# Symptoms: Cannot connect to database
# Check: Database URL format and credentials
python -c "
from app.core.database import engine
try:
    with engine.connect() as conn:
        print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database error: {e}')
"
```

#### Issue 3: SQLAlchemy Type Errors
```bash
# Symptoms: Column assignment errors
# Solution: Use helper functions (already implemented)
# Check: All services use safe_set_attr_value()
```

#### Issue 4: Memory Issues
```bash
# Monitor memory usage
pip install psutil

# Add memory monitoring
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
```

### Performance Optimization

#### Database Query Optimization
```python
# Use select loading for better performance
from sqlalchemy.orm import selectinload

predictions = db.query(Prediction)\
    .options(selectinload(Prediction.match))\
    .filter(Prediction.is_value_bet == True)\
    .all()
```

#### Caching Strategy
```python
# Implement Redis caching for expensive operations
import redis
from functools import wraps

redis_client = redis.from_url(settings.REDIS_URL)

def cache_result(expire_time=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire_time, json.dumps(result))
            return result
        return wrapper
    return decorator
```

---

## Maintenance Schedule

### Daily Tasks
- [ ] Check application logs for errors
- [ ] Monitor database performance
- [ ] Verify backup completion
- [ ] Check system resource usage

### Weekly Tasks  
- [ ] Update security patches
- [ ] Review performance metrics
- [ ] Clean up old log files
- [ ] Test backup restore process

### Monthly Tasks
- [ ] Update dependencies
- [ ] Review and rotate API keys
- [ ] Performance optimization review
- [ ] Security audit

---

*This deployment guide ensures a robust, secure, and maintainable production deployment of the Soccer prediction platform's FastAPI backend.*
