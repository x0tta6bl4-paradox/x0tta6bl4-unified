# 🐦 Canary Deployment Guide - x0tta6bl4 Unified Platform

## Обзор

Этот guide описывает процесс canary deployment для безопасного выпуска новых версий x0tta6bl4 Unified Platform в production.

## 🎯 Что такое Canary Deployment

Canary deployment - это стратегия развертывания, при которой новая версия приложения выпускается для ограниченного процента пользователей перед полным rollout.

### Преимущества
- **Минимизация риска**: Проблемы затрагивают только малую часть пользователей
- **Быстрое обнаружение**: Issues выявляются до полного развертывания
- **A/B тестирование**: Возможность сравнения метрик между версиями
- **Гибкость**: Легкий rollback при проблемах

## 🏗️ Архитектура

### Компоненты
- **Load Balancer**: Nginx или Istio для распределения трафика
- **Service Mesh**: Istio для продвинутого traffic management
- **Monitoring**: Prometheus + Grafana для отслеживания метрик
- **Feature Flags**: Для granular контроля функциональности

### Traffic Distribution
```
Production Traffic
        │
        ├── 5% → Canary Version (v2.1.0)
        │
        └── 95% → Stable Version (v2.0.0)
```

## 📋 Предварительные требования

### Инфраструктура
- Kubernetes кластер с Istio
- Horizontal Pod Autoscaler (HPA)
- Prometheus для метрик
- Service Mesh (Istio)

### Подготовка
```bash
# Создание canary namespace
kubectl create namespace x0tta6bl4-canary

# Настройка Istio для namespace
kubectl label namespace x0tta6bl4-canary istio-injection=enabled

# Создание service account для canary
kubectl create serviceaccount canary-deployer -n x0tta6bl4-canary
```

## 🚀 Процесс Canary Deployment

### Шаг 1: Подготовка Canary версии

#### Создание Canary Deployment
```yaml
# canary-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: x0tta6bl4-app-canary
  namespace: x0tta6bl4-canary
  labels:
    app: x0tta6bl4-app
    version: canary
spec:
  replicas: 2
  selector:
    matchLabels:
      app: x0tta6bl4-app
      version: canary
  template:
    metadata:
      labels:
        app: x0tta6bl4-app
        version: canary
        security.istio.io/tlsMode: istio
        service.istio.io/canonical-name: x0tta6bl4-app
    spec:
      containers:
      - name: app
        image: x0tta6bl4/x0tta6bl4-unified:canary
        ports:
        - containerPort: 8000
        env:
        - name: VERSION
          value: "canary"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Создание Canary Service
```yaml
# canary-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: x0tta6bl4-app-canary
  namespace: x0tta6bl4-canary
  labels:
    app: x0tta6bl4-app
    version: canary
spec:
  selector:
    app: x0tta6bl4-app
    version: canary
  ports:
  - name: http
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

### Шаг 2: Настройка Traffic Routing

#### Istio VirtualService для Canary
```yaml
# canary-virtualservice.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: x0tta6bl4-app
  namespace: x0tta6bl4
spec:
  http:
  - match:
    - headers:
        x-canary:
          exact: "true"
    route:
    - destination:
        host: x0tta6bl4-app-canary.x0tta6bl4-canary.svc.cluster.local
        subset: canary
  - route:  # Default routing
    - destination:
        host: x0tta6bl4-app-stable.x0tta6bl4.svc.cluster.local
        subset: stable
      weight: 95
    - destination:
        host: x0tta6bl4-app-canary.x0tta6bl4-canary.svc.cluster.local
        subset: canary
      weight: 5
```

#### DestinationRule для subsets
```yaml
# canary-destinationrule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: x0tta6bl4-app
  namespace: x0tta6bl4
spec:
  host: x0tta6bl4-app
  subsets:
  - name: stable
    labels:
      version: stable
  - name: canary
    labels:
      version: canary
```

### Шаг 3: Развертывание Canary

#### Применение конфигураций
```bash
# Развертывание canary версии
kubectl apply -f canary-deployment.yaml
kubectl apply -f canary-service.yaml
kubectl apply -f canary-virtualservice.yaml
kubectl apply -f canary-destinationrule.yaml

# Проверка развертывания
kubectl get pods -n x0tta6bl4-canary
kubectl get svc -n x0tta6bl4-canary
```

#### Масштабирование
```bash
# Автоматическое масштабирование для canary
kubectl autoscale deployment x0tta6bl4-app-canary \
  --cpu-percent=70 \
  --min=1 \
  --max=10 \
  -n x0tta6bl4-canary
```

## 📊 Мониторинг и Метрики

### Ключевые метрики для отслеживания

#### Application Metrics
```promql
# Response time comparison
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{version="canary"}[5m])) /
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{version="stable"}[5m]))

# Error rate comparison
rate(http_requests_total{status=~"5..", version="canary"}[5m]) /
rate(http_requests_total{version="canary"}[5m])

# CPU usage
rate(cpu_usage_seconds_total{version="canary"}[5m])
```

#### Business Metrics
```promql
# Conversion rate
rate(user_conversion_total{version="canary"}[5m])

# User engagement
rate(user_engagement_total{version="canary"}[5m])
```

### Grafana Dashboard

#### Создание Canary Dashboard
```json
{
  "dashboard": {
    "title": "Canary Deployment Monitoring",
    "panels": [
      {
        "title": "Response Time Comparison",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{version=~\"canary|stable\"}[5m]))",
            "legendFormat": "{{version}}"
          }
        ]
      },
      {
        "title": "Error Rate Comparison",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\", version=~\"canary|stable\"}[5m]) / rate(http_requests_total{version=~\"canary|stable\"}[5m])",
            "legendFormat": "{{version}}"
          }
        ]
      }
    ]
  }
}
```

## 🎛️ Управление Canary

### Gradual Traffic Increase

#### Автоматическое увеличение трафика
```bash
#!/bin/bash
# gradual_traffic_increase.sh

CURRENT_WEIGHT=5
TARGET_WEIGHT=50
STEP=5

while [ $CURRENT_WEIGHT -lt $TARGET_WEIGHT ]; do
  # Update VirtualService weight
  sed -i "s/weight: $CURRENT_WEIGHT/weight: $((CURRENT_WEIGHT + STEP))/" canary-virtualservice.yaml
  kubectl apply -f canary-virtualservice.yaml

  # Wait and monitor
  sleep 300  # 5 minutes

  # Check metrics
  if check_metrics_threshold; then
    echo "Metrics OK, increasing traffic to $((CURRENT_WEIGHT + STEP))%"
    CURRENT_WEIGHT=$((CURRENT_WEIGHT + STEP))
  else
    echo "Issues detected, stopping traffic increase"
    rollback_canary
    exit 1
  fi
done

echo "Canary deployment successful, proceeding to full rollout"
```

### Manual Traffic Control

#### Увеличение трафика на 10%
```bash
# Текущий VirtualService
kubectl get virtualservice x0tta6bl4-app -o yaml

# Обновление веса canary
kubectl patch virtualservice x0tta6bl4-app -n x0tta6bl4 --type merge -p '
spec:
  http:
  - route:
    - destination:
        host: x0tta6bl4-app-stable
      weight: 85
    - destination:
        host: x0tta6bl4-app-canary
      weight: 15
'
```

## 🚨 Обработка проблем

### Автоматический Rollback

#### Настройка Alert-based Rollback
```yaml
# canary-rollback-alert.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: canary-rollback-rules
  namespace: monitoring
spec:
  groups:
  - name: canary.rules
    rules:
    - alert: CanaryHighErrorRate
      expr: |
        rate(http_requests_total{status=~"5..", version="canary"}[5m]) /
        rate(http_requests_total{version="canary"}[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Canary version has high error rate"
        runbook_url: "https://docs.x0tta6bl4.com/runbooks/canary_rollback"

    - alert: CanarySlowResponse
      expr: |
        histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{version="canary"}[5m])) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Canary version has slow responses"
```

#### Автоматизированный Rollback
```bash
#!/bin/bash
# canary_rollback.sh

echo "Canary issues detected, initiating rollback..."

# Reduce canary traffic to 0
kubectl patch virtualservice x0tta6bl4-app -n x0tta6bl4 --type merge -p '
spec:
  http:
  - route:
    - destination:
        host: x0tta6bl4-app-stable
      weight: 100
'

# Scale down canary deployment
kubectl scale deployment x0tta6bl4-app-canary --replicas=0 -n x0tta6bl4-canary

# Notify team
curl -X POST $SLACK_WEBHOOK \
  -H 'Content-type: application/json' \
  -d '{"text": "🚨 Canary deployment rolled back due to issues"}'

echo "Canary rollback completed"
```

## ✅ Критерии успеха

### Технические критерии
- **Response Time**: < 500ms (не хуже stable версии более чем на 10%)
- **Error Rate**: < 1% (не выше stable версии более чем на 0.5%)
- **Resource Usage**: CPU/Memory в пределах 120% от stable версии
- **Availability**: 99.9% uptime

### Бизнес критерии
- **User Experience**: Не ухудшение ключевых метрик
- **Conversion Rate**: Не ниже stable версии более чем на 2%
- **Engagement**: Не ниже stable версии более чем на 5%

### Временные рамки
- **Initial Testing**: 30 минут - 2 часа
- **Traffic Increase**: 2-4 часа постепенного увеличения
- **Full Evaluation**: 24-48 часов мониторинга
- **Decision Point**: 48-72 часа от начала

## 📈 Продвинутые стратегии

### A/B Testing Integration

#### Feature Flags для Canary
```python
# Feature flag implementation
from featureflags import FeatureFlags

flags = FeatureFlags()

@app.route('/api/v1/process')
def process_data():
    if flags.is_enabled('new_algorithm', user_id=request.user.id):
        # Canary logic with new algorithm
        return process_with_new_algorithm(request.data)
    else:
        # Stable logic
        return process_with_stable_algorithm(request.data)
```

### Multi-Region Canary

#### Global Canary Deployment
```yaml
# multi-region-canary.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: x0tta6bl4-global
spec:
  http:
  - match:
    - sourceLabels:
        region: us-east
    route:
    - destination:
        host: x0tta6bl4-app-canary
      weight: 20
    - destination:
        host: x0tta6bl4-app-stable
      weight: 80
  - match:
    - sourceLabels:
        region: eu-west
    route:
    - destination:
        host: x0tta6bl4-app-canary
      weight: 10
    - destination:
        host: x0tta6bl4-app-stable
      weight: 90
  - route:  # Default for other regions
    - destination:
        host: x0tta6bl4-app-stable
```

## 📝 Документация и отчетность

### Canary Deployment Report
```
# Canary Deployment Report

Version: v2.1.0
Start Time: 2025-09-25 10:00 UTC
End Time: 2025-09-27 10:00 UTC

Traffic Distribution:
- Initial: 5%
- Peak: 50%
- Final: 100% (successful)

Metrics Comparison:
- Response Time: Stable: 450ms, Canary: 420ms (-6.7%)
- Error Rate: Stable: 0.8%, Canary: 0.6% (-25%)
- CPU Usage: Stable: 65%, Canary: 58% (-10.8%)

Business Impact:
- Conversion: +2.3%
- Engagement: +1.8%
- User Satisfaction: +5.1%

Decision: ✅ Proceed to full production rollout
```

### Lessons Learned
- [ ] Что работало хорошо
- [ ] Что можно улучшить
- [ ] Метрики для будущих canary deployments
- [ ] Процессные улучшения

## 🔧 Автоматизация

### CI/CD Pipeline Integration
```yaml
# .github/workflows/canary-deployment.yml
name: Canary Deployment

on:
  push:
    branches: [ main ]

jobs:
  canary-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2

    - name: Build and Push Canary Image
      run: |
        docker build -t x0tta6bl4/x0tta6bl4-unified:canary .
        docker push x0tta6bl4/x0tta6bl4-unified:canary

    - name: Deploy Canary
      run: |
        kubectl apply -f k8s/canary/
        kubectl wait --for=condition=available --timeout=300s deployment/x0tta6bl4-app-canary

    - name: Run Tests
      run: |
        npm test -- --testPathPattern=canary

    - name: Monitor and Decide
      run: |
        ./scripts/canary_monitor.sh

    - name: Rollback or Promote
      run: |
        if [ "$CANARY_STATUS" = "success" ]; then
          ./scripts/promote_canary.sh
        else
          ./scripts/rollback_canary.sh
        fi
```

---

**Версия**: 1.0
**Обновлено**: $(date)
**Контакт для обновлений**: DevOps Team