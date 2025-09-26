# 🚀 Quick Start Guide for Developers

## Welcome to x0tta6bl4 Unified!

This guide will get you up and running with the x0tta6bl4 Unified platform in minutes. Whether you're working on quantum computing, AI/ML, or enterprise features, this guide covers everything you need to start developing.

## Prerequisites

### System Requirements
- **Python**: 3.12 or higher
- **Docker**: Latest stable version
- **Docker Compose**: Latest version
- **Git**: Latest version
- **Kubernetes**: kubectl (for production deployment)

### Hardware Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **CPU**: Multi-core processor
- **Storage**: 20GB+ free space
- **GPU**: NVIDIA GPU (optional, for AI/ML development)

### Development Environment Setup

#### 1. Install Python 3.12+
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip

# macOS
brew install python@3.12

# Windows
# Download from python.org

# Verify installation
python3 --version  # Should show 3.12.x
pip --version
```

#### 2. Install Docker
```bash
# Ubuntu/Debian
sudo apt install docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER

# macOS
brew install --cask docker

# Windows
# Download Docker Desktop

# Verify installation
docker --version
docker-compose --version
```

#### 3. Install Development Tools
```bash
# Git (if not installed)
sudo apt install git

# Development tools
sudo apt install build-essential

# Verify
git --version
```

## Getting the Code

### Clone the Repository
```bash
# Clone the unified repository
git clone https://github.com/x0tta6bl4/x0tta6bl4-unified.git
cd x0tta6bl4-unified

# Verify structure
ls -la
```

### Repository Structure
```
x0tta6bl4-unified/
├── config/           # Configuration files
├── production/       # Production services
├── scripts/          # Utility scripts
├── tests/            # Test suites
├── docs/             # Documentation
├── main.py           # Main application entry
├── requirements.txt  # Python dependencies
├── docker-compose.yml # Docker services
└── Dockerfile        # Container definition
```

## Setting Up Development Environment

### 1. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Verify activation
which python  # Should show .venv/bin/python
```

### 2. Install Dependencies
```bash
# Install Python packages
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If exists

# Verify installation
pip list | grep -E "(fastapi|uvicorn|quantum|torch)"
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.development .env

# Edit configuration
nano .env  # or your preferred editor
```

#### Key Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/x0tta6bl4
REDIS_URL=redis://localhost:6379

# Quantum Computing
QUANTUM_BACKEND=ibm_quantum
IBM_QUANTUM_TOKEN=your_token_here

# AI/ML
CUDA_VISIBLE_DEVICES=0  # GPU device
MODEL_CACHE_DIR=./cache/models

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret

# Development
DEBUG=True
LOG_LEVEL=DEBUG
```

## Running the Platform

### Method 1: Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### Method 2: Local Development
```bash
# Activate virtual environment
source .venv/bin/activate

# Start main application
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Method 3: Kubernetes (Production)
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/base/
kubectl apply -f k8s/overlays/development/

# Check deployment
kubectl get pods
kubectl get services
```

## Accessing the Platform

### API Endpoints
```bash
# Main API
curl http://localhost:8000/health

# API Documentation (Swagger)
open http://localhost:8000/docs

# Alternative API docs
open http://localhost:8000/redoc
```

### Service URLs
- **API Gateway**: http://localhost:8000
- **Quantum Services**: http://localhost:8001
- **AI/ML Services**: http://localhost:8002
- **Enterprise Services**: http://localhost:8003

### Monitoring
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

## Development Workflow

### 1. Create Feature Branch
```bash
# Create and switch to feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit changes
git add .
git commit -m "Add your feature description"
```

### 2. Run Tests
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_api_gateway.py

# Run with coverage
python -m pytest --cov=src --cov-report=html
```

### 3. Code Quality Checks
```bash
# Lint code
flake8 src/
black src/
isort src/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

### 4. Submit Pull Request
```bash
# Push branch
git push origin feature/your-feature-name

# Create PR on GitHub
# Add description and link to issue
# Request review from team members
```

## Working with Different Components

### Quantum Computing Development

#### 1. Set Up Quantum Environment
```bash
# Install Qiskit
pip install qiskit qiskit-aer

# Configure IBM Quantum (optional)
# Get token from https://quantum-computing.ibm.com/
export IBM_QUANTUM_TOKEN=your_token
```

#### 2. Create Quantum Circuit
```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Create quantum circuit
qc = QuantumCircuit(2, 2)
qc.h(0)  # Hadamard gate
qc.cx(0, 1)  # CNOT gate
qc.measure_all()

# Simulate
simulator = AerSimulator()
job = simulator.run(transpile(qc, simulator), shots=1024)
result = job.result()
print(result.get_counts())
```

#### 3. Test Quantum Services
```bash
# Test quantum API
curl -X POST http://localhost:8001/quantum/execute \
  -H "Content-Type: application/json" \
  -d '{"circuit": "your_circuit_data"}'
```

### AI/ML Development

#### 1. Set Up AI Environment
```bash
# Install PyTorch/TensorFlow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# OR
pip install tensorflow

# Install transformers
pip install transformers accelerate
```

#### 2. Create ML Model
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Create and test model
model = SimpleModel()
input_tensor = torch.randn(1, 10)
output = model(input_tensor)
print(f"Output shape: {output.shape}")
```

#### 3. Test AI Services
```bash
# Test AI inference
curl -X POST http://localhost:8002/ai/inference \
  -H "Content-Type: application/json" \
  -d '{"model": "your_model", "input": [1, 2, 3, 4, 5]}'
```

### Enterprise Development

#### 1. Set Up Enterprise Environment
```bash
# Install enterprise dependencies
pip install sqlalchemy alembic fastapi-users

# Set up database
docker run -d --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 postgres:15
```

#### 2. Create API Endpoint
```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return {"users": users}

@router.post("/users")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(**user.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

#### 3. Test Enterprise Services
```bash
# Test user API
curl -X GET http://localhost:8003/users \
  -H "Authorization: Bearer your_token"

# Create user
curl -X POST http://localhost:8003/users \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "name": "Test User"}'
```

## Debugging and Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check what's using ports
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Or change port in config
export API_PORT=8001
```

#### Database Connection Issues
```bash
# Check database status
docker ps | grep postgres

# View database logs
docker logs postgres

# Connect to database
psql -h localhost -U postgres -d x0tta6bl4
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install missing packages
pip install missing-package

# Check virtual environment
which python
source .venv/bin/activate
```

#### Memory Issues
```bash
# Check memory usage
free -h

# Monitor processes
top -p $(pgrep -f "python|uvicorn")

# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory
```

### Logging and Monitoring

#### View Application Logs
```bash
# Docker logs
docker-compose logs -f api_gateway

# Application logs
tail -f logs/application.log

# Structured logging
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

#### Debug Mode
```bash
# Enable debug mode
export DEBUG=True
export LOG_LEVEL=DEBUG

# Restart services
docker-compose restart

# Check debug endpoints
curl http://localhost:8000/debug/info
```

## Testing Your Setup

### Health Checks
```bash
# Overall health
curl http://localhost:8000/health

# Individual services
curl http://localhost:8001/health  # Quantum
curl http://localhost:8002/health  # AI
curl http://localhost:8003/health  # Enterprise
```

### Integration Tests
```bash
# Run integration tests
python -m pytest tests/integration/ -v

# Test API endpoints
python -m pytest tests/test_api_gateway.py::test_health_check -v
```

### Performance Tests
```bash
# Load testing
ab -n 1000 -c 10 http://localhost:8000/health

# Quantum performance
python benchmarks/quantum_benchmark.py

# AI performance
python benchmarks/ai_benchmark.py
```

## Next Steps

### Learning Resources
1. **Architecture Overview**: Read `docs/architecture/overview.md`
2. **API Documentation**: Visit `http://localhost:8000/docs`
3. **Migration Guide**: See `docs/training/migration_scripts_guide.md`
4. **Troubleshooting**: Check `docs/training/troubleshooting_guide.md`

### Development Best Practices
1. **Code Style**: Follow PEP 8, use Black formatter
2. **Testing**: Write tests for all new features
3. **Documentation**: Document all public APIs
4. **Security**: Follow security guidelines
5. **Performance**: Profile and optimize code

### Getting Help
- **Documentation**: Check `docs/` directory
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Team**: Contact your team lead or architect

## Quick Commands Reference

```bash
# Development setup
git clone <repo>
cd x0tta6bl4-unified
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Running
docker-compose up -d          # Start services
python main.py               # Run locally
uvicorn main:app --reload    # Development server

# Testing
python -m pytest             # Run tests
python -m pytest --cov=src   # With coverage

# Code quality
black src/                   # Format code
flake8 src/                  # Lint code
mypy src/                    # Type check

# Debugging
docker-compose logs -f       # View logs
curl http://localhost:8000/health  # Health check
```

---

*Happy coding with x0tta6bl4 Unified! 🚀*

*Quick Start Guide - x0tta6bl4 Unified Training Materials*