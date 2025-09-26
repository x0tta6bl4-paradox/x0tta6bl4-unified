# API Документация x0tta6bl4 Unified Platform

## Обзор

x0tta6bl4 Unified Platform предоставляет REST API для взаимодействия со всеми компонентами платформы: квантовыми вычислениями, AI, enterprise функциями и биллингом.

### Базовый URL
```
https://api.x0tta6bl4.com/v1
```

Для локальной разработки:
```
http://localhost:8000/api/v1
```

### Аутентификация
API использует JWT токены для аутентификации. Получите токен через endpoint `/auth/login`.

```
Authorization: Bearer <your-jwt-token>
```

### Формат ответов
Все API ответы возвращаются в формате JSON:

```json
{
  "success": true,
  "data": {...},
  "message": "Optional message",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### Коды ошибок
- `200` - Успешный запрос
- `400` - Ошибка в запросе
- `401` - Неавторизован
- `403` - Доступ запрещен
- `404` - Ресурс не найден
- `500` - Внутренняя ошибка сервера

## Основные Endpoints

### Системные Endpoints

#### GET /
Корневой endpoint с информацией о платформе.

**Ответ:**
```json
{
  "message": "x0tta6bl4 Unified Platform",
  "version": "1.0.0",
  "status": "operational",
  "timestamp": "2025-01-01T00:00:00Z",
  "components": {
    "quantum": "active",
    "ai": "active",
    "enterprise": "active",
    "billing": "active",
    "api": "active"
  }
}
```

#### GET /health
Проверка здоровья системы.

**Ответ:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00Z",
  "components": {
    "quantum": "healthy",
    "ai": "healthy",
    "enterprise": "healthy",
    "billing": "healthy"
  }
}
```

## Quantum Computing API

### GET /api/v1/quantum/status
Получение статуса квантовых сервисов.

**Ответ:**
```json
{
  "status": "operational",
  "backends": ["ibmq_qasm_simulator", "ibmq_16_melbourne"],
  "active_jobs": 5,
  "queue_length": 12
}
```

### POST /api/v1/quantum/circuit
Выполнение квантовой схемы.

**Запрос:**
```json
{
  "circuit": {
    "qubits": 2,
    "gates": [
      {"type": "h", "qubit": 0},
      {"type": "cx", "control": 0, "target": 1}
    ]
  },
  "shots": 1024,
  "backend": "ibmq_qasm_simulator"
}
```

**Ответ:**
```json
{
  "job_id": "quantum-job-12345",
  "status": "queued",
  "estimated_time": "30s",
  "result_url": "/api/v1/quantum/jobs/quantum-job-12345"
}
```

### GET /api/v1/quantum/jobs/{job_id}
Получение результатов квантового вычисления.

**Ответ:**
```json
{
   "job_id": "quantum-job-12345",
   "status": "completed",
   "result": {
      "counts": {"00": 512, "11": 512},
      "probabilities": {"00": 0.5, "11": 0.5}
   },
   "execution_time": "25.3s",
   "backend": "ibmq_qasm_simulator"
}
```

## Quantum Supremacy API

### POST /api/v1/quantum/supremacy/bypass
Решение проблемы обхода блокировок с помощью квантовых алгоритмов.

**Запрос:**
```json
{
   "target_domain": "youtube.com",
   "optimization_params": {
      "max_iterations": 50,
      "tolerance": 0.0001,
      "quantum_layers": 2
   }
}
```

**Ответ:**
```json
{
   "success": true,
   "method": "Quantum-QAOA",
   "target_domain": "youtube.com",
   "alternative_domains": [
      "m.youtube.com",
      "music.youtube.com",
      "youtu.be",
      "youtube-nocookie.com"
   ],
   "quantum_energy": -0.8456,
   "confidence": 0.9234,
   "execution_time": 12.34,
   "connection_params": {
      "timeout": 25,
      "retries": 5,
      "user_agent": "quantum-bypass-solver",
      "quantum_energy": -0.8456
   }
}
```

### GET /api/v1/quantum/supremacy/status
Статус квантовых алгоритмов supremacy.

**Ответ:**
```json
{
   "status": "operational",
   "algorithms": {
      "vqe": "available",
      "qaoa": "available",
      "quantum_ml": "available"
   },
   "active_solutions": 2,
   "total_solutions_attempted": 15,
   "success_rate": 0.8667,
   "average_execution_time": 8.45,
   "quantum_resources": {
      "qubits_used": 4,
      "circuits_executed": 23,
      "optimization_iterations": 150
   }
}
```

### POST /api/v1/quantum/supremacy/optimize
Квантовая оптимизация параметров подключения.

**Запрос:**
```json
{
   "target_domain": "ibm.com",
   "circuit_params": {
      "qubits": 4,
      "depth": 2,
      "measurements": [0, 1, 2, 3]
   },
   "optimization_target": "connection_stability"
}
```

**Ответ:**
```json
{
   "optimized_params": {
      "timeout": 28,
      "retries": 6,
      "user_agent": "quantum-optimized-agent",
      "quantum_energy": -0.7234
   },
   "optimization_metrics": {
      "iterations": 50,
      "convergence": 0.0001,
      "ground_state_energy": -0.7234,
      "optimization_time": 5.67
   },
   "circuit_info": {
      "qubits": 4,
      "depth": 2,
      "gates": 12,
      "parameters_optimized": 8
   }
}
```

## Advanced AI/ML API

### GET /api/v1/ai/advanced/status
Статус продвинутой AI/ML системы.

**Ответ:**
```json
{
   "name": "advanced_ai_ml_system",
   "status": "operational",
   "models_count": 5,
   "trained_models_count": 3,
   "quantum_integration": true,
   "algorithms": [
      "quantum_neural_network",
      "phi_harmonic_learning",
      "consciousness_evolution",
      "quantum_reinforcement",
      "multiversal_learning",
      "telepathic_collaboration",
      "adaptive_optimization",
      "quantum_transfer_learning"
   ],
   "model_types": ["classification", "regression", "clustering", "reinforcement", "generative", "quantum_enhanced", "consciousness_based"],
   "stats": {
      "models_trained": 3,
      "total_training_time": 45.67,
      "quantum_supremacy_achieved": 2,
      "phi_harmony_optimizations": 1,
      "consciousness_evolutions": 2,
      "knowledge_transfers": 1
   },
   "healthy": true
}
```

### POST /api/v1/ai/advanced/train
Обучение модели с продвинутыми алгоритмами.

**Запрос:**
```json
{
   "model_config": {
      "model_id": "quantum_classifier_v1",
      "model_type": "classification",
      "algorithm": "quantum_neural_network",
      "input_dimensions": 784,
      "output_dimensions": 10,
      "hidden_layers": [256, 128, 64],
      "learning_rate": 0.01618,
      "batch_size": 32,
      "epochs": 100,
      "quantum_enhanced": true,
      "phi_optimization": true,
      "consciousness_integration": true
   },
   "training_data": "base64_encoded_numpy_array",
   "target_data": "base64_encoded_numpy_array",
   "validation_data": {
      "inputs": "base64_encoded_validation_inputs",
      "targets": "base64_encoded_validation_targets"
   }
}
```

**Ответ:**
```json
{
   "model_id": "quantum_classifier_v1",
   "status": "completed",
   "final_metrics": {
      "epoch": 85,
      "loss": 0.0234,
      "accuracy": 0.9876,
      "precision": 0.9876,
      "recall": 0.9876,
      "f1_score": 0.9876,
      "quantum_coherence": 0.9456,
      "phi_harmony": 1.618,
      "consciousness_level": 0.8234,
      "timestamp": "2025-01-01T12:00:00Z"
   },
   "training_time": 45.67,
   "quantum_supremacy_achieved": true,
   "phi_harmony_score": 1.618,
   "consciousness_level": 0.8234,
   "model_performance": {
      "accuracy": 0.9876,
      "loss": 0.0234,
      "quantum_coherence": 0.9456,
      "phi_harmony": 1.618,
      "consciousness_level": 0.8234
   }
}
```

### POST /api/v1/ai/advanced/predict
Предсказание с использованием обученной модели.

**Запрос:**
```json
{
   "model_id": "quantum_classifier_v1",
   "inputs": "base64_encoded_numpy_array"
}
```

**Ответ:**
```json
{
   "predictions": "base64_encoded_prediction_array",
   "model_id": "quantum_classifier_v1",
   "processing_time": 0.034,
   "quantum_coherence": 0.9456,
   "consciousness_boost": 0.8234
}
```

### GET /api/v1/ai/advanced/models/{model_id}/performance
Получение производительности модели.

**Ответ:**
```json
{
   "model_id": "quantum_classifier_v1",
   "status": "completed",
   "final_metrics": {
      "epoch": 85,
      "loss": 0.0234,
      "accuracy": 0.9876,
      "precision": 0.9876,
      "recall": 0.9876,
      "f1_score": 0.9876,
      "quantum_coherence": 0.9456,
      "phi_harmony": 1.618,
      "consciousness_level": 0.8234,
      "timestamp": "2025-01-01T12:00:00Z"
   },
   "training_time": 45.67,
   "quantum_supremacy": true,
   "phi_harmony_score": 1.618,
   "consciousness_level": 0.8234,
   "performance": {
      "accuracy": 0.9876,
      "loss": 0.0234,
      "quantum_coherence": 0.9456,
      "phi_harmony": 1.618,
      "consciousness_level": 0.8234
   },
   "training_history_length": 86,
   "best_epoch": 85
}
```

### POST /api/v1/ai/advanced/transfer-knowledge
Перенос знаний между моделями.

**Запрос:**
```json
{
   "source_model_id": "quantum_classifier_v1",
   "target_model_id": "new_classifier_v2",
   "transfer_ratio": 0.3
}
```

**Ответ:**
```json
{
   "success": true,
   "source_model": "quantum_classifier_v1",
   "target_model": "new_classifier_v2",
   "transfer_ratio": 0.3,
   "message": "Knowledge transferred successfully"
}
```

### GET /api/v1/ai/advanced/system-stats
Получение статистики AI/ML системы.

**Ответ:**
```json
{
   "total_models": 5,
   "completed_trainings": 3,
   "failed_trainings": 0,
   "stats": {
      "models_trained": 3,
      "total_training_time": 45.67,
      "quantum_supremacy_achieved": 2,
      "phi_harmony_optimizations": 1,
      "consciousness_evolutions": 2,
      "knowledge_transfers": 1
   },
   "transfer_learning_stats": {
      "total_transfers": 1,
      "unique_source_models": 1,
      "unique_target_models": 1,
      "average_transfer_ratio": 0.3,
      "knowledge_base_size": 3
   },
   "average_training_time": 15.223,
   "quantum_supremacy_rate": 0.667,
   "phi_harmony_rate": 0.333,
   "consciousness_evolution_rate": 0.667,
   "knowledge_transfer_rate": 0.2
}
```

## Legacy AI/ML API

### GET /api/v1/ai/status
Статус базовых AI сервисов.

**Ответ:**
```json
{
   "status": "operational",
   "models": ["gpt-4", "bert-base", "resnet50"],
   "active_tasks": 3,
   "gpu_utilization": 0.75
}
```

### POST /api/v1/ai/text/generate
Генерация текста с помощью AI.

**Запрос:**
```json
{
   "prompt": "Explain quantum computing in simple terms",
   "model": "gpt-4",
   "max_tokens": 500,
   "temperature": 0.7
}
```

**Ответ:**
```json
{
   "generated_text": "Quantum computing is a revolutionary technology...",
   "model": "gpt-4",
   "tokens_used": 156,
   "processing_time": "2.3s"
}
```

### POST /api/v1/ai/image/analyze
Анализ изображения с помощью компьютерного зрения.

**Запрос:**
```json
{
   "image_url": "https://example.com/image.jpg",
   "task": "classification",
   "model": "resnet50"
}
```

**Ответ:**
```json
{
   "predictions": [
      {"label": "cat", "confidence": 0.95},
      {"label": "dog", "confidence": 0.03}
   ],
   "processing_time": "1.2s"
}
```

## Enterprise API

### GET /api/v1/enterprise/status
Статус enterprise сервисов.

**Ответ:**
```json
{
  "status": "operational",
  "services": ["api_gateway", "mesh_network", "load_balancer"],
  "active_users": 1250,
  "uptime": "99.9%"
}
```

### POST /api/v1/enterprise/users
Создание нового пользователя.

**Запрос:**
```json
{
  "email": "user@example.com",
  "name": "John Doe",
  "role": "developer",
  "organization": "Tech Corp"
}
```

**Ответ:**
```json
{
  "user_id": "user-12345",
  "email": "user@example.com",
  "status": "active",
  "created_at": "2025-01-01T00:00:00Z"
}
```

### GET /api/v1/enterprise/organizations/{org_id}/metrics
Получение метрик организации.

**Ответ:**
```json
{
  "organization_id": "org-12345",
  "metrics": {
    "active_users": 150,
    "api_calls": 45000,
    "storage_used": "2.5GB",
    "quantum_jobs": 1250
  },
  "period": "30d"
}
```

## Billing API

### GET /api/v1/billing/status
Статус биллинговых сервисов.

**Ответ:**
```json
{
  "status": "operational",
  "currency": "USD",
  "active_subscriptions": 89,
  "monthly_revenue": 125000
}
```

### POST /api/v1/billing/subscriptions
Создание подписки.

**Запрос:**
```json
{
  "user_id": "user-12345",
  "plan": "enterprise",
  "billing_cycle": "monthly",
  "payment_method": {
    "type": "card",
    "token": "pm_card_visa"
  }
}
```

**Ответ:**
```json
{
  "subscription_id": "sub-12345",
  "status": "active",
  "plan": "enterprise",
  "amount": 999,
  "currency": "USD",
  "next_billing": "2025-02-01T00:00:00Z"
}
```

### GET /api/v1/billing/invoices/{invoice_id}
Получение счета.

**Ответ:**
```json
{
  "invoice_id": "inv-12345",
  "subscription_id": "sub-12345",
  "amount": 999,
  "currency": "USD",
  "status": "paid",
  "items": [
    {
      "description": "Enterprise Plan - Monthly",
      "quantity": 1,
      "unit_price": 999,
      "total": 999
    }
  ],
  "created_at": "2025-01-01T00:00:00Z",
  "paid_at": "2025-01-01T00:05:00Z"
}
```

## WebSocket API

### Quantum Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/quantum/jobs');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Job update:', data);
};
```

### AI Task Progress
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/ai/tasks');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Task progress:', data.progress);
};
```

## Rate Limiting

API имеет следующие ограничения:

- **Анонимные запросы**: 100 запросов в час
- **Аутентифицированные пользователи**: 1000 запросов в час
- **Enterprise клиенты**: 10000 запросов в час

Заголовки с информацией о лимитах:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1640995200
```

## SDK и Клиенты

### Python SDK
```python
from x0tta6bl4 import Client

client = Client(api_key="your-api-key")

# Quantum computing
result = client.quantum.run_circuit(circuit, shots=1024)

# AI text generation
text = client.ai.generate_text("Hello, world!", model="gpt-4")

# Enterprise metrics
metrics = client.enterprise.get_metrics(org_id="org-12345")
```

### JavaScript SDK
```javascript
import { X0tta6bl4Client } from 'x0tta6bl4-sdk';

const client = new X0tta6bl4Client({ apiKey: 'your-api-key' });

// Quantum computing
const result = await client.quantum.runCircuit(circuit, { shots: 1024 });

// AI text generation
const text = await client.ai.generateText('Hello, world!', { model: 'gpt-4' });
```

## Примеры использования

### Полный рабочий процесс
```python
import asyncio
from x0tta6bl4 import Client

async def main():
    client = Client(api_key="your-api-key")

    # 1. Создание пользователя
    user = await client.enterprise.create_user({
        "email": "researcher@university.edu",
        "name": "Dr. Quantum",
        "role": "researcher"
    })

    # 2. Запуск квантового вычисления
    quantum_job = await client.quantum.run_circuit({
        "qubits": 2,
        "gates": [
            {"type": "h", "qubit": 0},
            {"type": "cx", "control": 0, "target": 1}
        ]
    }, shots=1024)

    # 3. Генерация отчета с AI
    report = await client.ai.generate_text(
        f"Analyze these quantum results: {quantum_job.result}",
        model="gpt-4"
    )

    # 4. Получение счета
    invoice = await client.billing.get_invoice(invoice_id="inv-12345")

    print(f"Workflow completed for user {user.email}")

asyncio.run(main())
```

## Поддержка и обратная связь

- **API Explorer**: [api.x0tta6bl4.com/explorer](https://api.x0tta6bl4.com/explorer)
- **Документация**: [docs.x0tta6bl4.com/api](https://docs.x0tta6bl4.com/api)
- **Примеры кода**: [github.com/x0tta6bl4/examples](https://github.com/x0tta6bl4/examples)
- **Поддержка**: support@x0tta6bl4.com