#!/bin/bash
# Start Monitoring Stack for x0tta6bl4 Unified Platform
# Запуск стека мониторинга для x0tta6bl4 Unified Platform

set -e

echo "🚀 Starting x0tta6bl4 Monitoring Stack..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${RED}❌ Port $port is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $port is available${NC}"
        return 0
    fi
}

# Check required ports
echo "🔍 Checking required ports..."
check_port 9090 || exit 1  # Prometheus
check_port 9093 || exit 1  # Alertmanager
check_port 3000 || exit 1  # Grafana
check_port 8000 || exit 1  # Main metrics
check_port 8001 || exit 1  # Quantum metrics
check_port 8002 || exit 1  # AI metrics
check_port 8003 || exit 1  # Enterprise metrics
check_port 8004 || exit 1  # Billing metrics
check_port 9100 || exit 1  # Node Exporter

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ../data/prometheus
mkdir -p ../data/grafana
mkdir -p ../data/alertmanager
mkdir -p ../logs

# Start Node Exporter
echo "📊 Starting Node Exporter..."
docker run -d \
  --name x0tta6bl4-node-exporter \
  --network host \
  -v /proc:/host/proc:ro \
  -v /sys:/host/sys:ro \
  -v /:/rootfs:ro \
  prom/node-exporter \
  --path.procfs=/host/proc \
  --path.rootfs=/rootfs \
  --path.sysfs=/host/sys \
  --collector.filesystem.mount-points-exclude="^/(sys|proc|dev|host|etc)($$|/)"

# Start Prometheus
echo "📈 Starting Prometheus..."
docker run -d \
  --name x0tta6bl4-prometheus \
  --network host \
  -v $(pwd)/../prometheus.yml:/etc/prometheus/prometheus.yml:ro \
  -v $(pwd)/../alert_rules.yml:/etc/prometheus/alert_rules.yml:ro \
  -v $(pwd)/../data/prometheus:/prometheus \
  prom/prometheus \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/etc/prometheus/console_libraries \
  --web.console.templates=/etc/prometheus/consoles \
  --storage.tsdb.retention.time=200h \
  --web.enable-lifecycle

# Start Alertmanager
echo "🚨 Starting Alertmanager..."
docker run -d \
  --name x0tta6bl4-alertmanager \
  --network host \
  -v $(pwd)/../alertmanager.yml:/etc/alertmanager/config.yml:ro \
  -v $(pwd)/../data/alertmanager:/alertmanager \
  prom/alertmanager \
  --config.file=/etc/alertmanager/config.yml \
  --storage.path=/alertmanager

# Start Grafana
echo "📊 Starting Grafana..."
docker run -d \
  --name x0tta6bl4-grafana \
  --network host \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -v $(pwd)/../grafana/datasources:/etc/grafana/provisioning/datasources:ro \
  -v $(pwd)/../grafana/dashboards:/etc/grafana/provisioning/dashboards:ro \
  -v $(pwd)/../data/grafana:/var/lib/grafana \
  grafana/grafana

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Start custom metrics collectors
echo "🔧 Starting custom metrics collectors..."

# Main metrics collector
echo "📊 Starting main metrics collector..."
python3 collect_custom_metrics.py > ../logs/main_metrics.log 2>&1 &
echo $! > ../data/main_metrics.pid

# Quantum metrics collector
echo "🔬 Starting quantum metrics collector..."
python3 ../metrics/quantum_metrics.py > ../logs/quantum_metrics.log 2>&1 &
echo $! > ../data/quantum_metrics.pid

# AI metrics collector
echo "🤖 Starting AI metrics collector..."
python3 ../metrics/ai_metrics.py > ../logs/ai_metrics.log 2>&1 &
echo $! > ../data/ai_metrics.pid

# Enterprise metrics collector
echo "🏢 Starting enterprise metrics collector..."
python3 ../metrics/enterprise_metrics.py > ../logs/enterprise_metrics.log 2>&1 &
echo $! > ../data/enterprise_metrics.pid

# Billing metrics collector
echo "💰 Starting billing metrics collector..."
python3 ../metrics/billing_metrics.py > ../logs/billing_metrics.log 2>&1 &
echo $! > ../data/billing_metrics.pid

echo ""
echo -e "${GREEN}✅ x0tta6bl4 Monitoring Stack started successfully!${NC}"
echo ""
echo "🌐 Access URLs:"
echo "  📈 Prometheus:     http://localhost:9090"
echo "  🚨 Alertmanager:   http://localhost:9093"
echo "  📊 Grafana:        http://localhost:3000 (admin/admin)"
echo "  📋 Main Metrics:   http://localhost:8000/metrics"
echo "  🔬 Quantum Metrics: http://localhost:8001/metrics"
echo "  🤖 AI Metrics:     http://localhost:8002/metrics"
echo "  🏢 Enterprise Metrics: http://localhost:8003/metrics"
echo "  💰 Billing Metrics: http://localhost:8004/metrics"
echo ""
echo "📁 Log files: ../logs/"
echo "💾 Data directories: ../data/"
echo ""
echo "🛑 To stop monitoring stack, run: ./stop_monitoring.sh"