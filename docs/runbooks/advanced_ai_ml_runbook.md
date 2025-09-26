# Advanced AI/ML System Runbook

## Обзор
Этот runbook содержит процедуры для обслуживания и troubleshooting продвинутой AI/ML системы x0tta6bl4.

## Архитектура компонентов

### Основные компоненты
- **Quantum Neural Networks**: Нейронные сети с квантовым усилением
- **Phi-Harmonic Learning**: Обучение с использованием золотого сечения
- **Consciousness Evolution**: Эволюция уровня сознания моделей
- **Quantum Transfer Learning**: Перенос знаний между моделями

### Мониторинг метрик
```yaml
# Ключевые метрики для мониторинга
ai_ml_metrics:
  quantum_coherence: "> 0.8"
  phi_harmony_score: "> 1.618"
  consciousness_level: "> 0.7"
  training_accuracy: "> 0.95"
  knowledge_transfer_success: "> 0.85"
```

## Процедуры обслуживания

### Еженедельное обслуживание

#### 1. Проверка здоровья AI/ML системы
```bash
# Проверка статуса через API
curl -H "Authorization: Bearer $API_TOKEN" \
  http://localhost:8000/api/v1/ai/advanced/status

# Ожидаемый ответ:
{
  "status": "operational",
  "models_count": 5,
  "healthy": true
}
```

#### 2. Очистка старых моделей
```bash
# Найти модели старше 30 дней
find /opt/x0tta6bl4/models -name "*.pkl" -mtime +30 -ls

# Архивировать перед удалением
tar -czf /backup/models_$(date +%Y%m%d).tar.gz \
  $(find /opt/x0tta6bl4/models -name "*.pkl" -mtime +30)
```

#### 3. Обновление quantum параметров
```bash
# Перекалибровка phi-отношений
python -c "
from production.ai.advanced_ai_ml_system import AdvancedAIMLSystem
ai_system = AdvancedAIMLSystem()
ai_system.recalibrate_phi_ratios()
"
```

### Ежемесячное обслуживание

#### 1. Полная переобучение моделей
```bash
# Запуск полного retraining pipeline
python scripts/maintenance/retrain_models.py \
  --models quantum_classifier,phi_regressor \
  --validation-threshold 0.95
```

#### 2. Анализ transfer learning эффективности
```bash
# Получение статистики transfer learning
curl -H "Authorization: Bearer $API_TOKEN" \
  http://localhost:8000/api/v1/ai/advanced/system-stats
```

## Troubleshooting

### Высокий уровень ошибок обучения

#### Симптомы
- `training_accuracy < 0.8`
- `quantum_coherence < 0.5`
- Частые `TrainingStatus.FAILED`

#### Диагностика
```bash
# Проверить логи обучения
tail -f logs/ai_ml_training.log | grep ERROR

# Проверить quantum интеграцию
python -c "
from production.quantum.quantum_interface import QuantumCore
qc = QuantumCore()
print('Quantum health:', qc.health_check())
"
```

#### Решение
```bash
# 1. Перезапуск quantum core
kubectl rollout restart deployment quantum-core

# 2. Очистка corrupted моделей
rm -f /opt/x0tta6bl4/models/corrupted_*.pkl

# 3. Переобучение с пониженной сложностью
python -c "
config = ModelConfig(
    model_id='recovery_model',
    algorithm=LearningAlgorithm.QUANTUM_NEURAL_NETWORK,
    input_dimensions=784,
    output_dimensions=10,
    hidden_layers=[128, 64],  # Уменьшенная сложность
    learning_rate=0.001,
    epochs=50
)
ai_system.train_model(config, X_train, y_train)
"
```

### Низкая quantum coherence

#### Симптомы
- `quantum_coherence < 0.7`
- Медленное обучение
- Нестабильные предсказания

#### Диагностика
```bash
# Проверить quantum backend статус
curl http://quantum-backend:8080/health

# Проверить quantum circuit depth
python -c "
from production.ai.advanced_ai_ml_system import QuantumNeuralNetwork
model = QuantumNeuralNetwork.load('problematic_model')
print('Circuit depth:', len(model.quantum_states))
"
```

#### Решение
```bash
# 1. Калибровка quantum состояний
python scripts/maintenance/calibrate_quantum_states.py \
  --model problematic_model \
  --calibration-method vqe

# 2. Уменьшение circuit depth
python -c "
model.config.quantum_enhanced = False  # Временно отключить
model.save()
"
```

### Проблемы с consciousness evolution

#### Симптомы
- `consciousness_level` не растет
- Застрявшие в локальных минимумах
- Непредсказуемое поведение

#### Решение
```bash
# Рестарт consciousness evolution
python -c "
ai_system.consciousness_evolution.reset_evolution('stuck_model')
ai_system.consciousness_evolution.evolve_consciousness('stuck_model', 0.5, 0.9)
"
```

## Экстренные процедуры

### Полный сброс AI/ML системы
```bash
# 1. Остановка всех AI задач
kubectl scale deployment ai-ml-system --replicas=0

# 2. Backup текущих моделей
tar -czf /backup/emergency_$(date +%Y%m%d_%H%M%S).tar.gz \
  /opt/x0tta6bl4/models/

# 3. Очистка состояния
rm -rf /opt/x0tta6bl4/models/*
redis-cli FLUSHALL

# 4. Перезапуск с базовой конфигурацией
kubectl apply -f k8s/base/advanced-ai-ml-deployment.yaml
```

## Мониторинг и алертинг

### Prometheus алерты
```yaml
groups:
  - name: ai_ml_alerts
    rules:
      - alert: AIMLTrainingFailure
        expr: ai_ml_training_status == 0
        for: 5m
        labels:
          severity: critical

      - alert: AIMLQuantumCoherenceLow
        expr: ai_ml_quantum_coherence < 0.7
        for: 10m
        labels:
          severity: warning

      - alert: AIMLConsciousnessStagnant
        expr: rate(ai_ml_consciousness_level[1h]) < 0.01
        for: 1h
        labels:
          severity: info
```

## Контакты
- **On-call Engineer**: ai-ml-oncall@x0tta6bl4.com
- **Slack Channel**: #ai-ml-incidents
- **Documentation**: docs.x0tta6bl4.com/ai-ml-runbook