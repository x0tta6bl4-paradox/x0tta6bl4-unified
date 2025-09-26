# 🔧 Troubleshooting Guide

## Common Problems and Solutions

This guide covers the most common issues you'll encounter when working with x0tta6bl4 Unified and provides step-by-step solutions.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Database Problems](#database-problems)
- [API Issues](#api-issues)
- [Quantum Computing Problems](#quantum-computing-problems)
- [AI/ML Issues](#aiml-issues)
- [Performance Problems](#performance-problems)
- [Security Issues](#security-issues)
- [Migration Issues](#migration-issues)
- [Monitoring Problems](#monitoring-problems)

## Installation Issues

### Python Version Problems

**Problem**: `python3 --version` shows wrong version
```
Python 3.8.10
```

**Solution**:
```bash
# Install Python 3.12
sudo apt update
sudo apt install python3.12 python3.12-venv

# Set as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --set python3 /usr/bin/python3.12

# Verify
python3 --version  # Should show 3.12.x
```

### Virtual Environment Issues

**Problem**: Virtual environment not activating
```
source .venv/bin/activate
# No change in prompt
```

**Solutions**:
```bash
# Check if virtualenv exists
ls -la .venv/

# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

# Check activation
which python  # Should show .venv/bin/python
```

**Problem**: Packages not installing in virtualenv
```
pip install fastapi
# Installs globally instead of virtualenv
```

**Solution**:
```bash
# Ensure activation
source .venv/bin/activate

# Check pip location
which pip  # Should show .venv/bin/pip

# Reinstall pip in virtualenv
python -m pip install --upgrade pip
```

### Docker Installation Problems

**Problem**: Docker commands fail with permission denied
```
docker: Got permission denied while trying to connect to the Docker daemon socket
```

**Solutions**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart session or run:
newgrp docker

# Or run with sudo (not recommended for development)
sudo docker --version
```

**Problem**: Docker Compose version mismatch
```
ERROR: Version in "./docker-compose.yml" is unsupported
```

**Solution**:
```bash
# Check versions
docker --version
docker-compose --version

# Update Docker Compose
sudo apt remove docker-compose
sudo apt install docker-compose-plugin

# Or use docker compose (new syntax)
docker compose up -d
```

## Runtime Errors

### Import Errors

**Problem**: Module not found errors
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solutions**:
```bash
# Install missing package
pip install fastapi

# Check requirements.txt
pip install -r requirements.txt

# Check virtual environment
source .venv/bin/activate
pip list | grep fastapi
```

**Problem**: Circular import errors
```
ImportError: cannot import name 'X' from partially initialized module 'Y'
```

**Solutions**:
```bash
# Check import order in files
# Move imports to top of file
# Use TYPE_CHECKING for forward references

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .other_module import OtherClass
```

### Port Conflicts

**Problem**: Port already in use
```
[Errno 48] Address already in use
```

**Solutions**:
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Or change port in config
export API_PORT=8001
python main.py
```

**Problem**: Multiple services trying to use same port
```
Port 5432 is already allocated
```

**Solution**:
```bash
# Check Docker containers
docker ps

# Stop conflicting container
docker stop <container_name>

# Or modify docker-compose.yml ports
ports:
  - "5433:5432"  # Change host port
```

### Memory Issues

**Problem**: Out of memory errors
```
MemoryError: Unable to allocate array
```

**Solutions**:
```bash
# Check system memory
free -h

# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory

# For Linux, add to /etc/docker/daemon.json:
{
  "default-ulimits": {
    "memlock": -1
  }
}

# Restart Docker
sudo systemctl restart docker
```

## Database Problems

### Connection Issues

**Problem**: Database connection refused
```
psycopg2.OperationalError: could not connect to server
```

**Solutions**:
```bash
# Check if database is running
docker ps | grep postgres

# Check database logs
docker logs postgres

# Verify connection string
echo $DATABASE_URL

# Test connection
psql -h localhost -U user -d database
```

**Problem**: Authentication failed
```
FATAL: password authentication failed for user
```

**Solutions**:
```bash
# Check password in environment
echo $DATABASE_PASSWORD

# Reset password in Docker
docker exec -it postgres psql -U postgres -c "ALTER USER user PASSWORD 'new_password';"

# Update .env file
DATABASE_URL=postgresql://user:new_password@localhost:5432/database
```

### Migration Issues

**Problem**: Alembic migration fails
```
alembic.util.exc.CommandError: Can't locate revision
```

**Solutions**:
```bash
# Check migration status
alembic current

# List migrations
alembic history

# Generate new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

**Problem**: Database schema mismatch
```
sqlalchemy.exc.ProgrammingError: column does not exist
```

**Solution**:
```bash
# Reset database (development only!)
docker-compose down -v
docker-compose up -d postgres

# Re-run migrations
alembic upgrade head
```

## API Issues

### Endpoint Not Found

**Problem**: 404 errors on API calls
```
GET /api/v1/health - 404 Not Found
```

**Solutions**:
```bash
# Check if service is running
curl http://localhost:8000/health

# Check API routes
python -c "from main import app; print([route.path for route in app.routes])"

# Check service logs
docker-compose logs api_gateway
```

**Problem**: CORS errors in browser
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solutions**:
```bash
# Check CORS configuration in FastAPI
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Authentication Problems

**Problem**: JWT token invalid
```
401 Unauthorized: Invalid token
```

**Solutions**:
```bash
# Check token expiration
# Decode token at jwt.io

# Check secret key
echo $JWT_SECRET_KEY

# Regenerate token
curl -X POST http://localhost:8000/auth/login \
  -d "username=user&password=pass"
```

**Problem**: Permission denied
```
403 Forbidden: Insufficient permissions
```

**Solution**:
```bash
# Check user roles
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/auth/me

# Verify RBAC configuration
# Check role assignments in database
```

## Quantum Computing Problems

### Backend Connection Issues

**Problem**: Quantum backend unavailable
```
IBMQBackendError: Backend not found
```

**Solutions**:
```bash
# Check IBM Quantum token
echo $IBM_QUANTUM_TOKEN

# Verify backend name
# Available backends: ibmq_qasm_simulator, ibmq_16_melbourne, etc.

# Check network connectivity
ping quantum-computing.ibm.com
```

**Problem**: Job queue full
```
IBMQJobError: Job queue is full
```

**Solutions**:
```bash
# Use simulator instead
backend = provider.get_backend('ibmq_qasm_simulator')

# Wait and retry
import time
time.sleep(300)  # Wait 5 minutes

# Check queue status
backend.status()
```

### Circuit Compilation Errors

**Problem**: Circuit compilation fails
```
CircuitError: Invalid gate
```

**Solutions**:
```bash
# Check Qiskit version
pip show qiskit

# Validate circuit
from qiskit import transpile
try:
    transpiled = transpile(circuit, backend)
except Exception as e:
    print(f"Compilation error: {e}")

# Simplify circuit
# Remove unsupported gates
# Check qubit count vs backend capacity
```

## AI/ML Issues

### GPU Problems

**Problem**: CUDA not available
```
RuntimeError: CUDA is not available
```

**Solutions**:
```bash
# Check GPU status
nvidia-smi

# Install CUDA toolkit
# Follow NVIDIA CUDA installation guide

# Check PyTorch CUDA version
python -c "import torch; print(torch.cuda.is_available())"

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
```

**Problem**: GPU memory insufficient
```
CUDA out of memory
```

**Solutions**:
```bash
# Reduce batch size
batch_size = 16  # Instead of 32

# Use gradient accumulation
accumulation_steps = 4

# Clear GPU cache
torch.cuda.empty_cache()

# Use smaller model
model = SmallerModel()  # Instead of LargeModel
```

### Model Loading Issues

**Problem**: Model download fails
```
HTTPError: 403 Client Error
```

**Solutions**:
```bash
# Check internet connection
ping huggingface.co

# Set HuggingFace token
export HUGGINGFACE_TOKEN=your_token

# Use local cache
export TRANSFORMERS_CACHE=./cache/models
```

**Problem**: Model incompatible with dependencies
```
ImportError: incompatible version
```

**Solution**:
```bash
# Check versions
pip show torch transformers

# Update dependencies
pip install --upgrade torch transformers

# Or downgrade if needed
pip install torch==1.12.0
```

## Performance Problems

### Slow API Responses

**Problem**: API responses taking too long
```
Response time > 5 seconds
```

**Solutions**:
```bash
# Check database queries
# Add database indexes
CREATE INDEX idx_name ON table (column);

# Enable caching
from fastapi_cache import FastAPICache
FastAPICache.init(RedisBackend(redis), prefix="api")

# Profile code
python -m cProfile main.py
```

**Problem**: High CPU usage
```
CPU usage > 90%
```

**Solutions**:
```bash
# Check running processes
top -p $(pgrep -f python)

# Optimize code
# Use async/await
# Implement connection pooling
# Add rate limiting
```

### Memory Leaks

**Problem**: Memory usage keeps growing
```
Memory usage increases over time
```

**Solutions**:
```bash
# Check for object references
# Use weak references where appropriate

# Profile memory usage
from memory_profiler import profile
@profile
def my_function():
    pass

# Implement garbage collection
import gc
gc.collect()
```

## Security Issues

### SSL/TLS Problems

**Problem**: SSL certificate errors
```
ssl.SSLError: certificate verify failed
```

**Solutions**:
```bash
# Update certificates
sudo apt install ca-certificates
sudo update-ca-certificates

# Disable SSL verification (development only!)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Check certificate validity
openssl s_client -connect example.com:443
```

### Secret Management Issues

**Problem**: Secrets not loading
```
KeyError: SECRET_KEY
```

**Solutions**:
```bash
# Check .env file
cat .env

# Export environment variables
export SECRET_KEY=your_secret_key

# Use secret management service
# AWS Secrets Manager, Vault, etc.
```

## Migration Issues

### Script Failures

**Problem**: Migration script crashes
```
Traceback (most recent call last): ...
```

**Solutions**:
```bash
# Run with debug logging
python scripts/migration/migrate.py --dry-run -v

# Check source directories exist
ls -la /home/x0tta6bl4
ls -la /home/x0tta6bl4-next

# Check permissions
sudo chown -R $USER:$USER x0tta6bl4-unified/
```

**Problem**: Validation fails after migration
```
❌ Project structure: Отсутствуют директории: config
```

**Solutions**:
```bash
# Check migration logs
cat migration.log

# Manually create missing directories
mkdir -p config production

# Re-run validation
python scripts/migration/validation.py
```

### Rollback Problems

**Problem**: Rollback fails
```
❌ Резервная копия не найдена
```

**Solutions**:
```bash
# Find backup directory
find /home -name "*backup*" -type d

# Manually restore from backup
cp -r /path/to/backup/* x0tta6bl4-unified/

# Check backup integrity
ls -la x0tta6bl4-unified-backup-*/
```

## Monitoring Problems

### Metrics Not Collecting

**Problem**: Prometheus metrics empty
```
No data points found
```

**Solutions**:
```bash
# Check Prometheus configuration
cat prometheus.yml

# Verify targets are up
curl http://localhost:8000/metrics

# Check Prometheus logs
docker logs prometheus
```

**Problem**: Grafana dashboards blank
```
No data to display
```

**Solutions**:
```bash
# Check data source configuration
# Grafana UI → Configuration → Data Sources

# Verify Prometheus URL
http://prometheus:9090

# Import dashboards
# Grafana UI → Dashboards → Import
```

### Logging Issues

**Problem**: Logs not appearing
```
No log entries found
```

**Solutions**:
```bash
# Check log levels
export LOG_LEVEL=DEBUG

# Verify log file permissions
ls -la logs/

# Check logging configuration
cat config/logging.yaml

# Restart services
docker-compose restart
```

## Getting Help

### Diagnostic Commands

```bash
# System information
uname -a
python3 --version
docker --version

# Service status
docker-compose ps
docker stats

# Network connectivity
ping google.com
curl -I http://localhost:8000/health

# Log analysis
tail -f logs/application.log
grep ERROR migration.log
```

### Support Resources

1. **Documentation**: Check `docs/` directory
2. **GitHub Issues**: Search existing issues
3. **Team Chat**: Ask in development channels
4. **Logs**: Provide relevant log excerpts
5. **Environment**: Include system information

### Emergency Procedures

#### Service Down
```bash
# Quick restart
docker-compose restart

# Full rebuild
docker-compose down
docker-compose up --build -d

# Check health
curl http://localhost:8000/health
```

#### Data Loss
```bash
# Stop all services
docker-compose down

# Restore from backup
cp -r backup/* .

# Verify data integrity
python scripts/verify_data.py

# Restart services
docker-compose up -d
```

---

*Remember: When in doubt, check the logs first!*

*Troubleshooting Guide - x0tta6bl4 Unified Training Materials*