# CE Africa Valuation Platform - Deployment Guide

## 🚀 Quick Start: Recommended Deployment Options

### Option 1: Streamlit Cloud (Easiest - Recommended for MVP)

**Pros:**
- ✅ Free tier available
- ✅ Zero configuration
- ✅ Automatic HTTPS
- ✅ GitHub integration
- ✅ Built-in secrets management

**Steps:**
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select repository and branch
6. Set main file: `app_refactored.py`
7. Add secrets in Streamlit Cloud dashboard:
   ```
   [supabase]
   url = "your-supabase-url"
   anon_key = "your-anon-key"
   service_role_key = "your-service-role-key"
   
   [dev]
   user_id = "00000000-0000-0000-0000-000000000000"
   ```
8. Deploy!

**Cost:** Free (1 app) or $20/month (unlimited apps)

---

### Option 2: Railway (Best Balance)

**Pros:**
- ✅ Simple deployment
- ✅ PostgreSQL included
- ✅ GitHub integration
- ✅ Environment variables
- ✅ $5/month starter plan

**Steps:**
1. Sign up at [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway auto-detects Python/Streamlit
6. Add environment variables:
   ```bash
   STREAMLIT_SERVER_PORT=8501
   STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```
7. Add secrets (same as Streamlit Cloud)
8. Deploy!

**Cost:** $5/month (starter plan)

**Note:** Railway can also host PostgreSQL if you want to migrate from Supabase.

---

### Option 3: Render (Free Tier Available)

**Pros:**
- ✅ Free tier (with limitations)
- ✅ PostgreSQL add-on available
- ✅ GitHub integration
- ✅ Auto-deployments

**Cons:**
- ⚠️ Free tier apps sleep after 15 min inactivity
- ⚠️ Slower cold starts

**Steps:**
1. Sign up at [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect GitHub repository
4. Configure:
   - **Name:** ce-africa-valuation
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app_refactored.py --server.port=$PORT --server.address=0.0.0.0`
5. Add environment variables (same as above)
6. Deploy!

**Cost:** Free (with limitations) or $7/month (no sleep)

---

### Option 4: AWS App Runner (Production)

**Pros:**
- ✅ Enterprise-grade
- ✅ Auto-scaling
- ✅ High availability
- ✅ Full AWS integration

**Steps:**
1. Create `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "app_refactored.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Push to AWS ECR or GitHub
3. Create App Runner service in AWS Console
4. Configure environment variables
5. Deploy!

**Cost:** ~$25-50/month (pay-as-you-go)

---

### Option 5: Google Cloud Run (Production)

**Pros:**
- ✅ Serverless
- ✅ Auto-scaling
- ✅ Pay per use
- ✅ Good for variable traffic

**Steps:**
1. Create `Dockerfile` (same as AWS)
2. Build and push to Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/ce-africa-valuation
   ```
3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy ce-africa-valuation \
     --image gcr.io/PROJECT-ID/ce-africa-valuation \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```
4. Configure environment variables in Cloud Run console
5. Deploy!

**Cost:** ~$10-30/month (pay per use)

---

## 📋 Pre-Deployment Checklist

### 1. Code Preparation
- [ ] All code committed to Git
- [ ] `.gitignore` excludes sensitive files
- [ ] `requirements.txt` is up to date
- [ ] No hardcoded secrets in code
- [ ] All migrations tested locally

### 2. Database Preparation
- [ ] Production Supabase project created
- [ ] All migrations run on production database
- [ ] RLS policies configured
- [ ] Database backups enabled
- [ ] Connection pooling configured (if needed)

### 3. Environment Variables
- [ ] Supabase URL configured
- [ ] Supabase keys configured
- [ ] Any API keys configured
- [ ] Feature flags set (if any)

### 4. Testing
- [ ] Application runs locally
- [ ] All features tested
- [ ] Database connections work
- [ ] Error handling works
- [ ] Performance acceptable

---

## 🔐 Security Checklist

### Secrets Management
- [ ] No secrets in code
- [ ] Secrets stored in hosting platform
- [ ] Secrets rotated regularly
- [ ] Different secrets for dev/staging/prod

### Database Security
- [ ] RLS policies enabled
- [ ] Service role key only in backend
- [ ] Anon key used in frontend
- [ ] Database backups encrypted

### Application Security
- [ ] HTTPS enabled
- [ ] CORS configured correctly
- [ ] Input validation on all inputs
- [ ] SQL injection prevention
- [ ] Rate limiting (if applicable)

---

## 📊 Monitoring Setup

### Recommended Tools
1. **Sentry** - Error tracking
2. **Datadog/New Relic** - Application monitoring
3. **Supabase Dashboard** - Database monitoring
4. **Uptime Robot** - Uptime monitoring

### Key Metrics to Monitor
- Application uptime
- Response times
- Error rates
- Database query performance
- User activity

---

## 🔄 CI/CD Setup (Optional but Recommended)

### GitHub Actions Example

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          pytest tests/
      
      - name: Deploy to Streamlit Cloud
        # Streamlit Cloud auto-deploys on push
        # Or add deployment step for other platforms
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: App won't start**
- Check environment variables are set
- Verify Python version (3.11+)
- Check logs for errors

**Issue: Database connection fails**
- Verify Supabase URL and keys
- Check network/firewall settings
- Verify RLS policies allow access

**Issue: Slow performance**
- Enable connection pooling
- Optimize database queries
- Add caching where appropriate

**Issue: Secrets not working**
- Verify secret names match code
- Check secret format (TOML vs JSON)
- Ensure secrets are saved in hosting platform

---

## 📞 Support Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Supabase Docs:** https://supabase.com/docs
- **Railway Docs:** https://docs.railway.app
- **Render Docs:** https://render.com/docs

---

## 🎯 Recommended Deployment Path

### Phase 1: MVP (Week 1)
1. Deploy to **Streamlit Cloud** (free, fast)
2. Test with real users
3. Gather feedback

### Phase 2: Production (Month 1)
1. Migrate to **Railway** or **Render** (better performance)
2. Set up monitoring
3. Configure custom domain

### Phase 3: Scale (Month 3+)
1. Move to **AWS App Runner** or **GCP Cloud Run** (if needed)
2. Add auto-scaling
3. Implement advanced monitoring

---

**Last Updated:** December 2025
