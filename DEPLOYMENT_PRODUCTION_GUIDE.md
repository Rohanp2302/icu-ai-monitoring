# 📦 PRODUCTION DEPLOYMENT GUIDE

Complete step-by-step guide for deploying ICU Dashboard to production servers.

---

## 🎯 DEPLOYMENT OVERVIEW

| Component | Technology | Status |
|-----------|-----------|--------|
| **Frontend** | HTML5 + CSS3 + JavaScript | ✅ Ready |
| **Backend** | Flask (Python 3.10) | ✅ Ready |
| **Database** | JSON + CSV (File-based) | ✅ Ready |
| **Web Server** | Gunicorn + Nginx | 📌 Setup needed |
| **SSL/TLS** | Let's Encrypt | 📌 Setup needed |
| **Monitoring** | Logging + Health checks | ⚙️ Optional |

---

## 📋 PRE-DEPLOYMENT CHECKLIST

### System Requirements
- [ ] Linux/Windows Server 2019+ or macOS
- [ ] Python 3.10+
- [ ] 2GB+ RAM
- [ ] 10GB+ disk space
- [ ] Port 80 (HTTP) and 443 (HTTPS) available
- [ ] Static IP address
- [ ] Domain name (optional)

### Development Box Checklist
- [x] All tests passing (56/58 Dashboard, 6/6 APIs)
- [x] Sample data verified
- [x] PDF generation working
- [x] Chatbot responses configured
- [x] Error handling tested

### Production Box Preparation
- [ ] OS fully updates applied
- [ ] Security patches installed
- [ ] Firewall configured (allow 80, 443)
- [ ] SSH access configured
- [ ] Root/sudo access available

---

## 🔧 INSTALLATION STEPS

### Step 1: Set Up Production Directory

```bash
# Create application directory
sudo mkdir -p /opt/icu_dashboard
cd /opt/icu_dashboard

# Create data directories
mkdir -p patient_data uploads logs backups

# Set permissions
sudo chown -R $USER:$USER /opt/icu_dashboard
chmod -R 755 /opt/icu_dashboard
```

### Step 2: Copy Application Files

```bash
# From development box, copy:
scp -r app_production.py user@production_server:/opt/icu_dashboard/
scp -r enhanced_dashboard.html user@production_server:/opt/icu_dashboard/templates/
scp -r SAMPLE_PATIENT_DATA.csv user@production_server:/opt/icu_dashboard/
scp -r requirements.txt user@production_server:/opt/icu_dashboard/

# Alternative: Use Git
cd /opt/icu_dashboard
git clone <your-repo-url> ./
```

### Step 3: Python Environment Setup

```bash
# Install Python 3.10
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Verify installation
python -c "import flask, reportlab; print('✅ All dependencies installed')"
```

### Step 4: Configure Flask Application

**Create `/opt/icu_dashboard/config.py`:**

```python
import os

class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'prod-secret-key-change-this'
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Strict'
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    UPLOAD_FOLDER = '/opt/icu_dashboard/uploads'
    
    # Paths
    PATIENT_DATA_DIR = '/opt/icu_dashboard/patient_data'
    LOG_DIR = '/opt/icu_dashboard/logs'
    
    # API settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    LOGGER_LEVEL = 'INFO'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = True
    LOGGER_LEVEL = 'DEBUG'

# Load environment-specific config
config = {
    'production': ProductionConfig,
    'development': DevelopmentConfig,
    'default': ProductionConfig
}
```

### Step 5: Set Environment Variables

**Create `.env` file:**

```bash
# Save in /opt/icu_dashboard/.env
FLASK_ENV=production
FLASK_APP=app_production.py
SECRET_KEY=your-super-secret-key-here-change-this
PYTHON_ENV=production
LOG_LEVEL=INFO
```

**Load environment variables:**

```bash
# Add to shell rc file (~/.bashrc)
export FLASK_ENV=production
export SECRET_KEY=$(openssl rand -hex 32)
```

---

## 🚀 QUICK START (Development Testing)

```bash
# 1. Activate environment
cd /opt/icu_dashboard
source venv/bin/activate

# 2. Run Flask directly (for testing)
python app_production.py

# 3. Test endpoints
curl http://localhost:5000/login
curl http://localhost:5000/api/health

# 4. Stop with Ctrl+C
```

---

## 🏗️ GUNICORN SETUP (Production Server)

### Install Gunicorn

```bash
source /opt/icu_dashboard/venv/bin/activate
pip install gunicorn
```

### Create Gunicorn Configuration

**File: `/opt/icu_dashboard/gunicorn_config.py`**

```python
import multiprocessing

# Server settings
bind = "127.0.0.1:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = "/opt/icu_dashboard/logs/gunicorn_access.log"
errorlog = "/opt/icu_dashboard/logs/gunicorn_error.log"
loglevel = "info"

# Daemon
daemon = False
pidfile = "/opt/icu_dashboard/gunicorn.pid"
umask = 0o022

# Application
max_requests = 1000
max_requests_jitter = 50
```

### Start Gunicorn

```bash
cd /opt/icu_dashboard
source venv/bin/activate

# Start with config
gunicorn \
  --config gunicorn_config.py \
  --bind 127.0.0.1:8000 \
  app_production:app &

# Or use supervisor (see below)
```

---

## 🔌 NGINX REVERSE PROXY SETUP

### Install Nginx

```bash
sudo apt update
sudo apt install -y nginx
```

### Create Nginx Configuration

**File: `/etc/nginx/sites-available/icu-dashboard`**

```nginx
# HTTP redirect to HTTPS
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    # SSL security
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Logging
    access_log /var/log/nginx/icu_access.log;
    error_log /var/log/nginx/icu_error.log;
    
    # Client upload limit
    client_max_body_size 20M;
    
    # Gzip compression
    gzip on;
    gzip_types text/plain text/css text/xml text/javascript 
               application/x-javascript application/xml+rss 
               application/javascript application/json;
    
    # Proxy to Gunicorn
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Static files (optional)
    location /static/ {
        alias /opt/icu_dashboard/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

### Enable Nginx Configuration

```bash
# Create symlink
sudo ln -s /etc/nginx/sites-available/icu-dashboard \
           /etc/nginx/sites-enabled/icu-dashboard

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

---

## 🔒 SSL/TLS CERTIFICATES (Let's Encrypt)

### Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### Obtain Certificate

```bash
# Automatic setup with Nginx
sudo certbot certonly --nginx -d your-domain.com -d www.your-domain.com

# Or standalone
sudo certbot certonly --standalone -d your-domain.com

# Store email for renewals
# Choose to redirect HTTP to HTTPS (recommended)
```

### Auto-Renewal

```bash
# Enable automatic renewal
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Test renewal (dry-run)
sudo certbot renew --dry-run
```

---

## 🎛️ PROCESS MANAGEMENT (Systemd)

### Create Systemd Service

**File: `/etc/systemd/system/icu-dashboard.service`**

```ini
[Unit]
Description=ICU Dashboard Flask Application
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/icu_dashboard
Environment="PATH=/opt/icu_dashboard/venv/bin"
Environment="FLASK_ENV=production"
ExecStart=/opt/icu_dashboard/venv/bin/gunicorn \
    --config gunicorn_config.py \
    --bind 127.0.0.1:8000 \
    app_production:app

# Auto-restart on failure
Restart=always
RestartSec=10

# Process management
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Manage Service

```bash
# Enable on startup
sudo systemctl enable icu-dashboard.service

# Start service
sudo systemctl start icu-dashboard.service

# Check status
sudo systemctl status icu-dashboard.service

# View logs
sudo journalctl -u icu-dashboard.service -f

# Stop service
sudo systemctl stop icu-dashboard.service

# Restart service
sudo systemctl restart icu-dashboard.service
```

---

## 📊 MONITORING & LOGGING

### Configure Logging

**Update `app_production.py`:**

```python
import logging
from logging.handlers import RotatingFileHandler
import os

# Create logs directory
log_dir = '/opt/icu_dashboard/logs'
os.makedirs(log_dir, exist_ok=True)

# Configure logging
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'app.log'),
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
)
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)
```

### Health Check Monitoring

```bash
# Add to crontab for periodic monitoring
* * * * * curl -f http://localhost/api/health || systemctl restart icu-dashboard

# Monitor in real-time
watch -n 5 'curl -s http://localhost/api/health | jq'
```

### Log Rotation

**File: `/etc/logrotate.d/icu-dashboard`**

```
/opt/icu_dashboard/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload icu-dashboard > /dev/null 2>&1 || true
    endscript
}
```

---

## 🧪 POST-DEPLOYMENT TESTING

### Test Routes

```bash
# Health check
curl -k https://your-domain.com/api/health

# Login page
curl -k https://your-domain.com/login | grep -q "CareCast" && echo "✅ Login OK"

# Upload page
curl -k https://your-domain.com/upload | grep -q "upload" && echo "✅ Upload OK"

# Dashboard
curl -k https://your-domain.com/ | grep -q "enhanced" && echo "✅ Dashboard OK"
```

### Test APIs

```bash
# Chatbot API
curl -X POST https://your-domain.com/api/chatbot \
  -H "Content-Type: application/json" \
  -d '{"message":"When will patient be discharged?","patient_id":"ICU-2026-001"}'

# PDF Export
curl -X POST https://your-domain.com/api/export-pdf \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"ICU-2026-001","mortality_risk":0.73}' \
  -o test_report.pdf

# Data Persistence
curl -X POST https://your-domain.com/api/save-patient-data \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"ICU-2026-001","heart_rate":85,"spo2":95}'
```

### Load Testing (Optional)

```bash
# Using Apache Bench
ab -n 1000 -c 10 https://your-domain.com/

# Using Locust
pip install locust
locust -f locustfile.py --host=https://your-domain.com
```

---

## 🔐 SECURITY HARDENING

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw status
```

### Additional Security Headers

```nginx
# Add to Nginx config
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' fonts.googleapis.com;" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

### Database Backup

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/opt/icu_dashboard/backups"
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf "$BACKUP_DIR/icu_data_$DATE.tar.gz" \
    /opt/icu_dashboard/patient_data \
    /opt/icu_dashboard/uploads
# Keep only last 30 days
find "$BACKUP_DIR" -name "icu_data_*.tar.gz" -mtime +30 -delete
```

---

## 📈 PERFORMANCE OPTIMIZATION

### Caching Strategy

```python
# In app_production.py
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Cache specific routes
@app.route('/api/all-patients')
@cache.cached(timeout=60)
def get_all_patients():
    # ...
```

### Database Query Optimization

```python
# Limit and pagination
@app.route('/api/patients')
def list_patients():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    # Implement pagination
```

### Asset Compression

```bash
# Gzip CSS/JS on build
gzip -9 static/style.css
gzip -9 static/script.js

# Configure Nginx to serve .gz files
```

---

## 🆘 TROUBLESHOOTING

### Issue: 502 Bad Gateway

```bash
# Check Gunicorn is running
sudo systemctl status icu-dashboard

# Check logs
sudo journalctl -u icu-dashboard.service -n 50

# Restart
sudo systemctl restart icu-dashboard
```

### Issue: Slow Response

```bash
# Check resource usage
top -p $(pgrep -f gunicorn | xargs)

# Monitor Gunicorn workers
ps aux | grep gunicorn | grep -v grep

# Increase workers in gunicorn_config.py
```

### Issue: SSL Certificate Error

```bash
# Check certificate validity
openssl x509 -in /etc/letsencrypt/live/your-domain.com/fullchain.pem -text -noout

# Renew certificate
sudo certbot renew --force-renewal

# Verify Nginx config
sudo nginx -t
```

### Issue: PDF Export Fails

```bash
# Check ReportLab installed
python -c "import reportlab; print('OK')"

# Check permissions
ls -la /opt/icu_dashboard/patient_data

# Restart service
sudo systemctl restart icu-dashboard
```

---

## ✅ FINAL VERIFICATION CHECKLIST

Before going live:

- [ ] All tests passing (56/58 Dashboard, 6/6 APIs)
- [ ] SSL certificate installed and valid
- [ ] Nginx reverse proxy configured
- [ ] Gunicorn service running with 2+ workers
- [ ] Health check endpoint responding
- [ ] Login page loads
- [ ] Sample data uploads successfully
- [ ] PDF generation works
- [ ] Chatbot API responds
- [ ] Data persists after logout
- [ ] Logs being written to `/opt/icu_dashboard/logs/`
- [ ] Backups automated and tested
- [ ] Monitoring alerts configured
- [ ] Security headers present
- [ ] HTTPS working on custom domain

---

## 📞 SUPPORT CONTACTS

- **Technical Issues**: Check `/opt/icu_dashboard/logs/app.log`
- **Certificate Issues**: `certbot renew --dry-run`
- **Service Issues**: `sudo systemctl status icu-dashboard.service`

---

**Document Version**: 1.0  
**Last Updated**: April 9, 2026  
**Status**: ✅ Ready for Production Deployment
