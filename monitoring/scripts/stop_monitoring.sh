#!/bin/bash
# Stop Monitoring Stack for x0tta6bl4 Unified Platform
# Остановка стека мониторинга для x0tta6bl4 Unified Platform

echo "🛑 Stopping x0tta6bl4 Monitoring Stack..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Stop custom metrics collectors
echo "🔧 Stopping custom metrics collectors..."

if [ -f ../data/main_metrics.pid ]; then
    kill $(cat ../data/main_metrics.pid) 2>/dev/null || true
    rm ../data/main_metrics.pid
    echo -e "${GREEN}✅ Main metrics collector stopped${NC}"
fi

if [ -f ../data/quantum_metrics.pid ]; then
    kill $(cat ../data/quantum_metrics.pid) 2>/dev/null || true
    rm ../data/quantum_metrics.pid
    echo -e "${GREEN}✅ Quantum metrics collector stopped${NC}"
fi

if [ -f ../data/ai_metrics.pid ]; then
    kill $(cat ../data/ai_metrics.pid) 2>/dev/null || true
    rm ../data/ai_metrics.pid
    echo -e "${GREEN}✅ AI metrics collector stopped${NC}"
fi

if [ -f ../data/enterprise_metrics.pid ]; then
    kill $(cat ../data/enterprise_metrics.pid) 2>/dev/null || true
    rm ../data/enterprise_metrics.pid
    echo -e "${GREEN}✅ Enterprise metrics collector stopped${NC}"
fi

if [ -f ../data/billing_metrics.pid ]; then
    kill $(cat ../data/billing_metrics.pid) 2>/dev/null || true
    rm ../data/billing_metrics.pid
    echo -e "${GREEN}✅ Billing metrics collector stopped${NC}"
fi

# Stop Docker containers
echo "🐳 Stopping Docker containers..."

docker stop x0tta6bl4-grafana 2>/dev/null || echo -e "${YELLOW}⚠️ Grafana container not running${NC}"
docker stop x0tta6bl4-alertmanager 2>/dev/null || echo -e "${YELLOW}⚠️ Alertmanager container not running${NC}"
docker stop x0tta6bl4-prometheus 2>/dev/null || echo -e "${YELLOW}⚠️ Prometheus container not running${NC}"
docker stop x0tta6bl4-node-exporter 2>/dev/null || echo -e "${YELLOW}⚠️ Node Exporter container not running${NC}"

# Remove Docker containers
echo "🗑️ Removing Docker containers..."
docker rm x0tta6bl4-grafana 2>/dev/null || true
docker rm x0tta6bl4-alertmanager 2>/dev/null || true
docker rm x0tta6bl4-prometheus 2>/dev/null || true
docker rm x0tta6bl4-node-exporter 2>/dev/null || true

echo -e "${GREEN}✅ x0tta6bl4 Monitoring Stack stopped successfully!${NC}"