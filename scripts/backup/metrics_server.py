#!/usr/bin/env python3
"""
Simple HTTP server to serve Prometheus metrics for backup service
Простой HTTP сервер для предоставления метрик Prometheus для сервиса backup
"""

import http.server
import socketserver
import os
import time
from urllib.parse import urlparse, parse_qs

class MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/metrics'):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()

            metrics_file = '/backups/metrics.prom'
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        content = f.read()
                        self.wfile.write(content.encode('utf-8'))
                except Exception as e:
                    self.wfile.write(f'# Error reading metrics: {e}\n'.encode('utf-8'))
            else:
                # Default metrics if file doesn't exist
                self.wfile.write(b'# Backup service metrics\n')
                self.wfile.write(b'backup_service_up 1\n')
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found\n')

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def main():
    port = 9091
    with socketserver.TCPServer(("", port), MetricsHandler) as httpd:
        print(f"Metrics server running on port {port}")
        httpd.serve_forever()

if __name__ == "__main__":
    main()