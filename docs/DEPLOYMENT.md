# Deployment Guide

This guide covers deploying SQL Query Buddy to production environments.

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Access to OpenAI API (for full functionality)
- Production database (PostgreSQL/MySQL recommended)

## Local Deployment

### Development Environment

```bash
# Clone and setup
git clone <repo-url>
cd SqlQueryBuddy

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest

# Run the app
python -m src.main
```

### Production Environment

1. **Set up environment**:
```bash
python -m venv venv_prod
source venv_prod/bin/activate
pip install -r requirements.txt
```

2. **Configure production settings**:
```bash
cp .env.example .env
# Edit .env with production settings:
# - OPENAI_API_KEY
# - DATABASE_URL (use PostgreSQL)
# - DEBUG=false
# - GRADIO_SHARE=false
```

3. **Create production database**:
```bash
# For PostgreSQL
createdb sql_query_buddy
# Configure connection string in .env
```

4. **Run with production server**:
```bash
# Using gunicorn with gradio
pip install gunicorn
gunicorn src.app:demo.app --bind 0.0.0.0:7860 --workers 4
```

## Docker Deployment

### Build Image

```bash
docker build -t sql-query-buddy:latest .
```

### Run Container

```bash
docker run -p 7860:7860 \
  -e OPENAI_API_KEY="sk-..." \
  -e DATABASE_URL="postgresql://..." \
  -e DEBUG=false \
  sql-query-buddy:latest
```

### Docker Compose

```bash
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Cloud Deployment

### AWS EC2

1. **Launch instance** (Ubuntu 20.04 LTS)
2. **Install dependencies**:
```bash
sudo apt update && sudo apt install -y python3.9 python3-pip python3-venv git
```

3. **Clone and setup**:
```bash
git clone <repo-url>
cd SqlQueryBuddy
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Configure .env with RDS database**:
```bash
DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/sql_query_buddy
```

5. **Run with systemd service**:
Create `/etc/systemd/system/sql-query-buddy.service`:
```ini
[Unit]
Description=SQL Query Buddy
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/SqlQueryBuddy
ExecStart=/home/ubuntu/SqlQueryBuddy/venv/bin/python -m src.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable sql-query-buddy
sudo systemctl start sql-query-buddy
```

### Heroku

1. **Create Procfile**:
```
web: python -m src.main
```

2. **Create runtime.txt**:
```
python-3.9.0
```

3. **Deploy**:
```bash
heroku create sql-query-buddy
git push heroku main
heroku config:set OPENAI_API_KEY="sk-..."
heroku open
```

### HuggingFace Spaces (Recommended for Gradio Apps)

**URL:** https://huggingface.co/spaces/rsprasanna/SqlQueryBuddy

HuggingFace Spaces is the recommended deployment platform for Gradio applications.

#### Initial Setup

1. **Create a new Space** at https://huggingface.co/new-space
   - Name: SqlQueryBuddy
   - SDK: Gradio
   - Hardware: CPU basic (free) or GPU if needed

2. **Push your code**:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SqlQueryBuddy
git push hf main
```

#### Setting Environment Variables (CRITICAL for API Keys)

**NEVER commit API keys to git!** Use HuggingFace Secrets instead:

1. **Go to your Space settings**:
   - Navigate to https://huggingface.co/spaces/YOUR_USERNAME/SqlQueryBuddy/settings
   - Click on "Repository secrets" section

2. **Add secrets** (one at a time):
   - Click "Add a secret"
   - Name: `OPENAI_API_KEY`
   - Value: `sk-proj-YOUR-ACTUAL-KEY-HERE`
   - Click "Add secret"

3. **Add other environment variables** as secrets:
   ```
   OPENAI_MODEL=gpt-4
   DEBUG=false
   DATABASE_TYPE=sqlite
   VECTOR_DB_TYPE=faiss
   SIMILARITY_THRESHOLD=0.7
   ```

4. **The app will automatically restart** and pick up the new secrets

#### Updating Your API Key

If your API key is exposed or needs rotation:

1. **Revoke the old key** in OpenAI dashboard: https://platform.openai.com/api-keys
2. **Generate a new key** in OpenAI
3. **Update the secret in HuggingFace**:
   - Go to Space settings → Repository secrets
   - Click "Edit" on OPENAI_API_KEY
   - Paste the new key
   - Click "Update secret"
4. **Space will auto-restart** with the new key (no code changes needed)

#### Monitoring Your Space

- **Logs**: Click "Logs" tab in your Space to see runtime output
- **Usage**: Check "Analytics" for visitor stats
- **Rebuilds**: Space rebuilds automatically on git push

#### Important Notes

- ✅ HuggingFace Spaces supports Docker (Dockerfile is included)
- ✅ Free CPU tier available (7860 port auto-configured)
- ✅ Secrets are encrypted and never exposed in logs
- ⚠️ SQLite database resets on each restart (use PostgreSQL for persistence)
- ⚠️ Free tier has limited resources (consider upgrading for production)

### Google Cloud Run

1. **Build and push image**:
```bash
docker build -t gcr.io/PROJECT_ID/sql-query-buddy:latest .
docker push gcr.io/PROJECT_ID/sql-query-buddy:latest
```

2. **Deploy**:
```bash
gcloud run deploy sql-query-buddy \
  --image gcr.io/PROJECT_ID/sql-query-buddy:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY="sk-..." \
  --port 7860
```

## Reverse Proxy Setup

### Nginx Configuration

```nginx
upstream sql_query_buddy {
    server localhost:7860;
}

server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://sql_query_buddy;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL/TLS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

## Database Setup

### PostgreSQL

```sql
-- Create database
CREATE DATABASE sql_query_buddy;

-- Create user
CREATE USER app_user WITH PASSWORD 'secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE sql_query_buddy TO app_user;
```

Connection string:
```
postgresql://app_user:secure_password@db-host:5432/sql_query_buddy
```

### MySQL

```sql
CREATE DATABASE sql_query_buddy;
CREATE USER 'app_user'@'localhost' IDENTIFIED BY 'secure_password';
GRANT ALL PRIVILEGES ON sql_query_buddy.* TO 'app_user'@'localhost';
FLUSH PRIVILEGES;
```

Connection string:
```
mysql+pymysql://app_user:secure_password@db-host:3306/sql_query_buddy
```

## Performance Optimization

1. **Enable query caching**:
```python
# In src/app.py, add caching layer for frequently used queries
```

2. **Database optimization**:
```sql
-- Create indexes on commonly filtered columns
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
```

3. **Use connection pooling**:
```python
# SQLAlchemy automatically handles this
engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=40)
```

4. **Enable logging**:
```bash
DEBUG=false LOG_LEVEL=INFO  # Reduced logging overhead
```

## Monitoring

### Health Check Endpoint

Add to `src/app.py`:
```python
@app.route("/health")
def health():
    return {"status": "healthy"}
```

### Log Monitoring

```bash
# View application logs
journalctl -u sql-query-buddy -f

# Centralized logging (with ELK stack)
# Configure in docker-compose.yml
```

### Alerting

- Monitor CPU and memory usage
- Alert on database connection failures
- Track API rate limit errors
- Monitor response times

## Backup and Recovery

```bash
# Database backup
pg_dump -h host -U user -d sql_query_buddy > backup.sql

# Restore
psql -h host -U user -d sql_query_buddy < backup.sql

# Docker volume backup
docker cp sql-query-buddy:/app/data ./backup/
```

## Security Checklist

- [ ] OPENAI_API_KEY is not in version control
- [ ] DEBUG mode is disabled in production
- [ ] Database credentials are secured
- [ ] HTTPS/TLS is enabled
- [ ] SQL injection tests pass
- [ ] Database is not publicly accessible
- [ ] Regular security updates applied
- [ ] Error messages don't expose sensitive info
- [ ] Rate limiting is configured
- [ ] Backups are tested

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Container won't start | Check logs: `docker logs <container>` |
| Database connection fails | Verify DATABASE_URL and network connectivity |
| Out of memory | Reduce worker processes or increase instance size |
| Slow queries | Check database indexes and query plans |
| High API costs | Implement query caching and result limits |

## Scaling

For high traffic:

1. **Horizontal scaling**: Run multiple instances behind load balancer
2. **Caching layer**: Add Redis for query result caching
3. **Async processing**: Use Celery for long-running operations
4. **Database**: Use read replicas for analytics queries
5. **Vector DB**: Migrate from in-memory to Pinecone/Weaviate for RAG

---

For more information, see [README.md](../README.md)
