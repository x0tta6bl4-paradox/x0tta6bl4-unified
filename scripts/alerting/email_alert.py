#!/usr/bin/env python3
"""
Simple Email Alerting System for x0tta6bl4
Бесплатная альтернатива для alerting без использования платных сервисов
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailAlerter:
    """Simple email alerting system"""

    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.alert_email = os.getenv('ALERT_EMAIL', 'admin@localhost')

    def send_alert(self, subject: str, message: str, severity: str = 'info'):
        """Отправка email алерта"""
        try:
            # Создание сообщения
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = self.alert_email
            msg['Subject'] = f"[{severity.upper()}] x0tta6bl4 Alert: {subject}"

            # Добавление timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            body = f"""
x0tta6bl4 Unified Platform Alert

Timestamp: {timestamp}
Severity: {severity.upper()}
Subject: {subject}

Message:
{message}

---
This is an automated alert from x0tta6bl4 monitoring system.
Please check the system status at: http://your-server:3000
            """

            msg.attach(MIMEText(body, 'plain'))

            # Отправка email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            text = msg.as_string()
            server.sendmail(self.smtp_username, self.alert_email, text)
            server.quit()

            logger.info(f"Alert sent successfully: {subject}")

        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")

def send_system_alert(subject: str, message: str, severity: str = 'warning'):
    """Удобная функция для отправки системных алертов"""
    alerter = EmailAlerter()
    alerter.send_alert(subject, message, severity)

def check_and_alert_system_status():
    """Проверка статуса системы и отправка алертов при необходимости"""
    try:
        import psutil
        import requests

        alerts_sent = []

        # Проверка CPU
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 80:
            send_system_alert(
                "High CPU Usage Detected",
                f"CPU usage is {cpu_usage}%. System may be overloaded.",
                "warning"
            )
            alerts_sent.append("CPU")

        # Проверка памяти
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            send_system_alert(
                "High Memory Usage Detected",
                f"Memory usage is {memory.percent}%. System may run out of memory.",
                "critical"
            )
            alerts_sent.append("Memory")

        # Проверка дискового пространства
        disk = psutil.disk_usage('/')
        if disk.percent > 85:
            send_system_alert(
                "Low Disk Space",
                f"Disk usage is {disk.percent}%. Only {disk.free // (1024**3)}GB free.",
                "warning"
            )
            alerts_sent.append("Disk")

        # Проверка API здоровья (если доступно)
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code != 200:
                send_system_alert(
                    "Application Health Check Failed",
                    f"Health check returned status {response.status_code}",
                    "critical"
                )
                alerts_sent.append("Health")
        except:
            send_system_alert(
                "Application Unreachable",
                "Cannot connect to application health endpoint",
                "critical"
            )
            alerts_sent.append("Connectivity")

        if alerts_sent:
            logger.info(f"System alerts sent for: {', '.join(alerts_sent)}")
        else:
            logger.info("System status OK - no alerts needed")

    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        send_system_alert(
            "Monitoring System Error",
            f"Failed to check system status: {str(e)}",
            "critical"
        )

if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)

    # Проверка системы
    check_and_alert_system_status()

    # Пример отправки тестового алерта
    # send_system_alert("Test Alert", "This is a test alert from x0tta6bl4 monitoring", "info")