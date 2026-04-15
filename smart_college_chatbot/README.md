# 🎓 Smart College Chatbot v3 — SR University

An intelligent NLP-powered chatbot for SR University, Warangal using **Sentence-BERT** embeddings for high-accuracy intent classification (90%+ accuracy).

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env and set a strong SECRET_KEY

# 4. Train the model
python train_model.py --evaluate

# 5. Run the server
python app.py
```

Visit http://localhost:5000

## 🧠 NLP Architecture (v3)

- **Primary**: Sentence-BERT (`all-MiniLM-L6-v2`) + cosine similarity
- **Fallback 1**: TF-IDF (word + char n-grams) + CalibratedLinearSVC
- **Fallback 2**: Keyword matching (no ML dependencies)

### Key Improvements
- BERT embeddings capture semantic meaning, not just word overlap
- Data augmentation with synonym substitution + question templates
- Train/test split on **original** patterns (no data leakage)
- Preprocessor output is properly used in the pipeline

## 🔒 Security Fixes Applied

| Issue | Severity | Fix |
|-------|----------|-----|
| Hardcoded secret key | Critical | Random key generation, env var required |
| SHA-256 passwords (no salt) | Critical | bcrypt with per-user salt |
| SQL injection in LIKE | Critical | Escaped `%` and `_` metacharacters |
| Wildcard CORS | Critical | Configurable allowed origins |
| No rate limiting on /chat | Major | 60 req/min per IP |
| session_id leaked in response | Major | Removed from JSON body |
| Preprocessor output unused | Major | Integrated into pipeline |
| Data leakage in evaluation | Major | Split before augmentation |
| commit-after-read bug | Major | Separate DB connections |
| No input sanitization (XSS) | Major | HTML escaping on all inputs |
| Credentials logged | Major | Password removed from log |
| random.seed(0) deterministic | Major | Instance-based RNG |
| chat_id None in feedback | Minor | Validated as non-null int |
| NLTK download at import | Minor | Lazy loading |
| Synonym replacement order | Minor | Single-pass compiled regex |
| Confidence thresholds mismatch | Minor | Centralised constants |
| No CSRF protection | Minor | Token-based CSRF for admin |
| Keyword score formula | Minor | Simple ratio without doubling |

## 📁 Project Structure

```
├── app.py                    # Flask application factory
├── train_model.py            # Training & evaluation script
├── requirements.txt          # Python dependencies
├── backend/
│   ├── nlp/
│   │   ├── preprocessor.py   # Text cleaning & tokenization
│   │   └── classifier.py     # BERT-based intent classifier
│   ├── models/
│   │   └── database.py       # SQLite operations (bcrypt auth)
│   └── routes/
│       ├── chat.py           # Chat endpoint (rate-limited)
│       ├── admin.py          # Admin panel (CSRF-protected)
│       └── api.py            # Public API
├── data/
│   ├── intents.json          # Training data (20+ intents)
│   └── schema.py             # DB schema
├── frontend/
│   ├── templates/            # HTML templates
│   └── static/               # CSS & JS
└── tests/
    └── test_chatbot.py       # Unit tests
```

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 📊 Expected Performance

With BERT embeddings:
- Training accuracy: ~98-100%
- Hold-out accuracy: ~90-95%
- Evaluation test accuracy: ~90%+

## ⚙️ Admin Panel

Visit `/admin` (default: admin/admin123). **Change the password immediately!**

## 📄 License

MIT
