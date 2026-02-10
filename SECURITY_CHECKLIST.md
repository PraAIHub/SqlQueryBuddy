# Security Checklist - SQL Query Buddy

Use this checklist before deploying to production.

## 🔴 CRITICAL (Must Complete Before Production)

### Secrets & Configuration
- [ ] Remove all `.env` files from repository
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` exists with template only
- [ ] Use AWS Secrets Manager / HashiCorp Vault in production
- [ ] All secrets loaded from environment variables only
- [ ] No hardcoded API keys, passwords, or tokens

### HTTPS/TLS
- [ ] Domain has valid SSL certificate
- [ ] HTTPS enforced (redirect HTTP → HTTPS)
- [ ] TLS version 1.2 or higher
- [ ] Certificate will be auto-renewed
- [ ] HSTS header enabled

### Database Security
- [ ] Database user has minimal permissions (SELECT only)
- [ ] Database connection uses SSL/TLS
- [ ] Connection string doesn't expose credentials
- [ ] Database backups are encrypted
- [ ] Database audit logging enabled

### API Security
- [ ] Authentication implemented for API endpoints
- [ ] Rate limiting configured (e.g., 60 requests/min)
- [ ] CORS properly configured (not `*`)
- [ ] Request validation middleware in place
- [ ] Response headers secured (no info leakage)

### Code
- [ ] All tests passing (22/22)
- [ ] Code review completed
- [ ] No hardcoded credentials in code
- [ ] All dependencies pinned to specific versions
- [ ] No eval() or exec() usage
- [ ] SQL injection tests passing (10/10)

---

## 🟠 HIGH PRIORITY (Complete Before Launch)

### Logging & Monitoring
- [ ] Structured logging implemented
- [ ] Error logs don't contain sensitive data
- [ ] Logs aggregated to central system
- [ ] Monitoring alerts configured
- [ ] Health check endpoint functional
- [ ] Performance metrics tracked

### Access Control
- [ ] User authentication required
- [ ] Role-based access control (RBAC) implemented
- [ ] API keys issued for service accounts
- [ ] API keys stored securely
- [ ] API keys rotated regularly

### Data Protection
- [ ] PII data minimization (only necessary data stored)
- [ ] Data retention policy defined
- [ ] Data deletion mechanism implemented
- [ ] GDPR/CCPA compliance checked
- [ ] User consent captured where needed

### Incident Response
- [ ] Incident response plan documented
- [ ] Security contact information published
- [ ] Vulnerability disclosure policy in place
- [ ] Incident response team trained
- [ ] Automated alerts for suspicious activity

---

## 🟡 MEDIUM PRIORITY (Complete Within 30 Days)

### Testing
- [ ] Penetration testing scheduled
- [ ] Security scanning integrated in CI/CD
- [ ] Dependency vulnerability scanning enabled
- [ ] OWASP Top 10 testing completed
- [ ] Load testing for DoS resistance

### Documentation
- [ ] Security architecture documented
- [ ] Threat model created
- [ ] Security policies documented
- [ ] Runbook for security incidents
- [ ] Disaster recovery plan documented

### Infrastructure
- [ ] Firewall rules configured
- [ ] Network segmentation implemented
- [ ] DDoS protection enabled
- [ ] WAF rules configured
- [ ] IP whitelisting for admin access

### Dependencies
- [ ] All dependencies updated to latest safe versions
- [ ] Dependency security scanning continuous
- [ ] Update policy established (weekly/monthly)
- [ ] Testing required before dependency updates
- [ ] Rollback plan for failed updates

---

## 🟢 LOW PRIORITY (Complete Within 90 Days)

### Advanced Security
- [ ] Bug bounty program considered
- [ ] Security training completed by team
- [ ] Third-party security audit scheduled
- [ ] Key rotation schedule established
- [ ] Disaster recovery tested

### Compliance
- [ ] SOC 2 compliance audit
- [ ] ISO 27001 certification path
- [ ] Industry-specific compliance checked
- [ ] Data privacy agreements in place
- [ ] Audit logs available for compliance

### Operations
- [ ] Regular security training schedule
- [ ] Patch management process
- [ ] Vulnerability disclosure handled
- [ ] Security metrics tracked
- [ ] Annual security review scheduled

---

## ✅ Pre-Deployment Verification

Run these commands before deploying:

```bash
# 1. Check for secrets
grep -r "sk-\|password\|api_key" src/ --include="*.py"
# Should find nothing

# 2. Run all tests
pytest tests/ -v
# Should show: 22 passed

# 3. Security audit
python -c "
from src.components.sql_generator import SQLValidator
tests = [
    ('SELECT * FROM users', True),
    ('DROP TABLE users', False),
    (\"'; DROP TABLE users; --\", False),
]
for query, should_pass in tests:
    is_valid, _ = SQLValidator.validate(query)
    assert is_valid == should_pass, f'Failed: {query}'
print('✅ All security tests passed')
"

# 4. Check dependencies
pip check
# Should show: No broken requirements found

# 5. Build Docker image
docker build -t sql-query-buddy:prod .
# Should succeed

# 6. Verify environment template
test -f .env.example && echo "✅ .env.example exists"
grep ".env" .gitignore && echo "✅ .env in .gitignore"
```

---

## 🚨 Security Incident Response

If you find a security vulnerability:

1. **Do NOT** commit the fix to public repo
2. **DO** create a private security fix branch
3. **DO** test thoroughly in staging
4. **DO** coordinate deployment carefully
5. **DO** notify users if data affected
6. **DO** document the incident

---

## 📋 Configuration Examples

### Secure Database URL
```python
# ❌ WRONG - Credentials in URL
DATABASE_URL=postgresql://user:password@host:5432/db

# ✅ CORRECT - Credentials in environment
DATABASE_URL=postgresql://user:${DB_PASSWORD}@host:5432/db
# Set DB_PASSWORD separately
```

### Secure Secrets Manager
```python
import boto3
client = boto3.client('secretsmanager')
secret = client.get_secret_value(SecretId='sql-query-buddy')
api_key = secret['SecretString']
```

### Rate Limiting
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/query")
@limiter.limit("60/minute")
async def query(request: Request):
    pass
```

### CORS Security
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)
```

---

## 📞 Security Contacts

- **Security Report**: security@yourdomain.com
- **On-call Security**: [Phone Number]
- **Incident Commander**: [Contact Info]

---

## 📚 Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [SANS Top 25](https://www.sans.org/top25-software-errors/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

**Last Updated**: February 10, 2026
**Status**: Ready for Contest Submission
**Next Review**: After deployment to production
