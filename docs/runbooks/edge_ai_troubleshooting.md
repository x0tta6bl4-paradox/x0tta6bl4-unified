# 🌐 Edge AI Troubleshooting Guide - x0tta6bl4 Unified Platform

## Распространенные edge AI проблемы и решения

### 1. Edge device connectivity failure
**Симптомы**: Edge устройства не отвечают на ping, потеряна связь
**Решение**:
```bash
# Проверить connectivity к edge устройствам
for device in edge-device-{01..10}; do
    ping -c 3 $device.x0tta6bl4.local
done

# Проверить VPN tunnel
ipsec statusall | grep edge-vpn

# Перезапустить edge connectivity service
systemctl restart edge-connectivity

# Проверить edge gateway
curl -f http://edge-gateway.x0tta6bl4.local/health
```

### 2. Edge AI model deployment failure
**Симптомы**: Модель не загружается на edge устройство
**Решение**:
```bash
# Проверить edge device storage
ssh edge-device-01 "df -h /opt/edge-ai/models"

# Validate model file integrity
ssh edge-device-01 "sha256sum /opt/edge-ai/models/latest_model.onnx"

# Проверить model compatibility
ssh edge-device-01 "python3 validate_edge_model.py --model latest_model.onnx"

# Redeploy model
./scripts/deploy/edge_model_deploy.sh --device edge-device-01 --force

# Проверить edge AI logs
ssh edge-device-01 "tail -f /var/log/edge-ai/deployment.log"
```

### 3. Edge inference performance degradation
**Симптомы**: Медленный inference, высокая latency на edge
**Решение**:
```bash
# Проверить edge device resources
ssh edge-device-01 "top -b -n1 | head -20"

# Monitor edge inference latency
curl -w "@edge_curl_format.txt" -o /dev/null -s http://edge-device-01.x0tta6bl4.local/inference

# Проверить model optimization
ssh edge-device-01 "python3 check_model_optimization.py"

# Clear edge cache
ssh edge-device-01 "rm -rf /tmp/edge_inference_cache/*"

# Restart edge inference service
ssh edge-device-01 "systemctl restart edge-ai-inference"
```

### 4. Edge device overheating
**Симптомы**: Edge устройства перегреваются, автоматическое отключение
**Решение**:
```bash
# Проверить temperature sensors
ssh edge-device-01 "sensors | grep Core"

# Проверить cooling system
ssh edge-device-01 "systemctl status edge-cooling"

# Adjust inference load
ssh edge-device-01 "python3 adjust_inference_load.py --reduce 50"

# Проверить thermal throttling
ssh edge-device-01 "dmesg | grep -i thermal"

# Emergency cooling procedure
ssh edge-device-01 "./emergency_cooling.sh"
```

### 5. Edge data synchronization failure
**Симптомы**: Edge устройства не синхронизируют данные с центральным сервером
**Решение**:
```bash
# Проверить sync status
curl -s http://edge-gateway.x0tta6bl4.local/sync/status | jq '.devices[] | select(.status != "synced")'

# Проверить network bandwidth
ssh edge-device-01 "iperf3 -c central-server.x0tta6bl4.local -t 10"

# Validate data integrity
ssh edge-device-01 "python3 validate_edge_data.py --check-integrity"

# Force sync
./scripts/sync/force_edge_sync.sh --device edge-device-01

# Проверить sync logs
ssh edge-device-01 "tail -f /var/log/edge-ai/sync.log"
```

### 6. Edge AI model accuracy degradation
**Симптомы**: Низкая accuracy inference на edge устройствах
**Решение**:
```bash
# Проверить model drift
ssh edge-device-01 "python3 detect_model_drift.py --baseline central --current edge"

# Retrain edge model
./scripts/training/edge_model_retraining.sh --device edge-device-01

# Update model weights
scp updated_model_weights.pkl edge-device-01:/opt/edge-ai/models/
ssh edge-device-01 "systemctl restart edge-ai-inference"

# Проверить calibration
ssh edge-device-01 "python3 calibrate_edge_model.py"
```

### 7. Edge device power management issues
**Симптомы**: Edge устройства не переходят в sleep mode, высокий power consumption
**Решение**:
```bash
# Проверить power management settings
ssh edge-device-01 "cat /sys/power/state"

# Adjust power profile
ssh edge-device-01 "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"

# Проверить battery status (если применимо)
ssh edge-device-01 "upower -i /org/freedesktop/UPower/devices/battery_BAT0"

# Reset power management
ssh edge-device-01 "systemctl restart edge-power-management"

# Monitor power consumption
ssh edge-device-01 "powertop --time=10"
```

### 8. Edge security compromise
**Симптомы**: Подозрительная активность, unauthorized access к edge устройствам
**Решение**:
```bash
# Проверить security logs
ssh edge-device-01 "journalctl -u edge-security --since '1 hour ago'"

# Scan for malware
ssh edge-device-01 "clamscan /opt/edge-ai/ --log=/var/log/clamav/scan.log"

# Rotate edge certificates
./scripts/security/rotate_edge_certificates.sh --device edge-device-01

# Update security policies
scp updated_security_policy.json edge-device-01:/etc/edge-ai/security/
ssh edge-device-01 "systemctl reload edge-security"

# Isolate compromised device
iptables -A INPUT -s edge-device-01 -j DROP
```

## Edge AI Incident Response

### Шаги при edge AI инциденте:
1. **Оценка**: Определить влияние на edge inference и data processing
2. **Сдерживание**: Изолировать affected edge устройства
3. **Восстановление**: Вернуть edge AI в рабочее состояние
4. **Анализ**: Определить причину edge failure
5. **Документация**: Записать edge AI lessons learned

### Edge AI Escalation:
- **P1 (Critical)**: Полная потеря edge connectivity, immediate response < 15 мин
- **P2 (High)**: Массовый edge inference failure, response < 1 час
- **P3 (Medium)**: Индивидуальные edge device issues, response < 4 часа
- **P4 (Low)**: Edge performance degradation, response по графику

## Автоматизированные Edge AI Diagnostics

### Edge Health Check Script
```bash
#!/bin/bash
# edge_health_check.sh

echo "=== Edge AI System Health Check ==="

# Check edge devices connectivity
for device in $(cat /etc/edge-devices.list); do
    if ping -c 1 -W 2 $device > /dev/null; then
        echo "✅ $device: CONNECTED"
    else
        echo "❌ $device: DISCONNECTED"
    fi
done

# Check edge inference performance
for device in $(cat /etc/edge-devices.list); do
    LATENCY=$(curl -w "%{time_total}" -o /dev/null -s http://$device.x0tta6bl4.local/inference 2>/dev/null)
    if (( $(echo "$LATENCY < 0.1" | bc -l) )); then
        echo "✅ $device inference: $LATENCY s"
    else
        echo "❌ $device inference: $LATENCY s (SLOW)"
    fi
done

# Check edge storage
for device in $(cat /etc/edge-devices.list); do
    USAGE=$(ssh $device "df /opt/edge-ai | tail -1 | awk '{print \$5}' | sed 's/%//'")
    if [ "$USAGE" -lt 80 ]; then
        echo "✅ $device storage: ${USAGE}%"
    else
        echo "❌ $device storage: ${USAGE}% (HIGH)"
    fi
done

echo "=== Health Check Complete ==="
```

### Edge Performance Monitoring
```bash
# Real-time edge metrics
watch -n 30 'curl -s http://edge-gateway.x0tta6bl4.local/metrics | jq ".edge_devices[] | {name: .name, inference_latency: .inference_latency, storage_usage: .storage_usage}"'

# Edge device temperature monitoring
for device in $(cat /etc/edge-devices.list); do
    ssh $device "sensors | grep -A 2 'Core 0'" &
done

# Edge network monitoring
iperf3 -c edge-gateway.x0tta6bl4.local -t 60 -i 10
```

## Edge AI Rollback Procedures

### Model Rollback
```bash
# Rollback to previous edge model version
ssh edge-device-01 "cp /opt/edge-ai/models/previous_model.onnx /opt/edge-ai/models/current_model.onnx"
ssh edge-device-01 "systemctl restart edge-ai-inference"
```

### Configuration Rollback
```bash
# Rollback edge configuration
ssh edge-device-01 "git checkout HEAD~1 -- /etc/edge-ai/"
ssh edge-device-01 "systemctl reload edge-ai-inference"
```

### Firmware Rollback
```bash
# Rollback edge firmware
ssh edge-device-01 "./firmware_rollback.sh --version previous"
ssh edge-device-01 "reboot"
```

## Edge AI Monitoring Integration

### Prometheus Edge Metrics
```yaml
# edge_metrics.yml
edge_device_connectivity: gauge
edge_inference_latency: histogram
edge_model_accuracy: gauge
edge_storage_usage: gauge
edge_temperature: gauge
edge_power_consumption: gauge
```

### Alert Rules
```yaml
# edge_alerts.yml
- alert: EdgeDeviceDisconnected
  expr: edge_device_connectivity == 0
  for: 5m
  labels:
    severity: critical

- alert: EdgeInferenceHighLatency
  expr: histogram_quantile(0.95, rate(edge_inference_latency_bucket[5m])) > 0.5
  for: 2m
  labels:
    severity: warning

- alert: EdgeStorageFull
  expr: edge_storage_usage > 90
  for: 10m
  labels:
    severity: critical

- alert: EdgeOverheating
  expr: edge_temperature > 80
  for: 2m
  labels:
    severity: critical
```

## Контакты

### Edge Computing Team
- **Техническая поддержка**: edge-support@x0tta6bl4.com
- **On-call инженер**: edge-oncall@x0tta6bl4.com
- **Архитекторы**: edge-architecture@x0tta6bl4.com

### Edge AI Team
- **ML инженеры**: edge-ai-ml@x0tta6bl4.com
- **DevOps**: edge-devops@x0tta6bl4.com

### Security Team
- **Edge security**: edge-security@x0tta6bl4.com
- **IoT security**: iot-security@x0tta6bl4.com

## Процедуры эскалации

### Edge P1 Incident
1. Немедленное уведомление Edge Computing Lead (SMS + Call)
2. Активация Edge Emergency Response Team
3. Изоляция affected edge устройств
4. Параллельная работа над восстановлением connectivity

### Edge P2 Incident
1. Уведомление Edge AI Team Lead
2. Оценка количества affected edge устройств
3. Gradual rollout of fixes to edge devices
4. Monitoring recovery progress

### Edge P3 Incident
1. Уведомление Edge Computing Team
2. Оценка влияния на edge inference
3. Individual device troubleshooting
4. Documentation of root cause

---

**Версия**: 1.0
**Обновлено**: $(date)
**Контакт для обновлений**: Edge Computing Team