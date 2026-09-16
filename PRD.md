# Product Requirements Document: House Price Prediction System

## 1. Executive Summary

**Product Name:** House Price Prediction System
**Version:** 1.0.0
**Status:** Draft

An ML-powered system that estimates residential property values based on structural, locational, and conditional features. Targeted at real estate agents, homeowners, buyers, and fintech platforms. Deployed via Flask/Gunicorn behind Nginx reverse proxy with rate limiting, security headers, and CSP.

## 2. Problem Statement

Property valuation is subjective, time-consuming, and varies widely between appraisers. There is no fast, data-driven way to get a reliable price estimate without a professional appraisal.

## 3. Goals & Success Metrics

| Metric | Target | Actual |
|---|---|---|
| Test R² Score | >= 0.85 | **0.996** |
| Test MAE | <= ₹50,000 | **~₹6.8 Lakhs** |
| Prediction latency (API) | < 500ms | **<100ms** |
| Model retrain time | < 5 min | **~6 min (all 3)** |

## 4. User Personas

- **Home Buyer:** Needs a fair-market estimate before making an offer.
- **Home Seller:** Wants to set a competitive listing price.
- **Real Estate Agent:** Needs quick valuations for client consultations.

## 5. Functional Requirements

### FR-1: Data Ingestion
- Load CSV datasets with property features
- Validate schema and handle missing/invalid values
- Support for 250K+ records across 20 Indian states, 42 cities

### FR-2: Data Preprocessing
- Remove duplicates and outliers (0.2% / 99.8% percentile clipping)
- Standardize numeric features via MinMaxScaler
- Split data into train/val/test (70/10/20)

### FR-3: Feature Engineering
- Floor ratio, top/ground/mid-floor indicators
- Sqft-per-BHK, BHK×size interaction
- Amenity counts, school/hospital ratio
- Age categories (new/old), squared age, decade built
- Log transforms (size, price-per-sqft)

### FR-4: Model Training
- Support multiple algorithms: XGBoost, LightGBM (GPU), Random Forest
- GPU-accelerated training (LightGBM on NVIDIA GPU)
- CLI arguments for single-model training or full comparison
- Output: serialized model (.joblib) + metrics JSON + multi-modal scores

### FR-5: Prediction API
- REST endpoint: `POST /predict` (rate-limited, validated)
- Accept JSON with 18 property features
- Return predicted price range (lower/upper bounds with confidence level)
- Multi-modal ensemble scoring (composite R² + MAE + MAPE weighting)
- Health check: `GET /health`
- Model inventory: `GET /models`
- State/city lookup: `GET /api/states`, `GET /api/cities`
- City rate lookup: `GET /api/rate`
- Security headers: CSP, X-Frame-Options, X-XSS-Protection, X-Content-Type-Options, Cache-Control, Referrer-Policy

### FR-6: Evaluation
- Metrics: MAE, MSE, RMSE, R², MAPE
- Per-dataset reporting (train/val/test)
- Multi-modal composite scoring for model selection

## 6. Non-Functional Requirements

- **Performance:** API responds within 500ms (typical <100ms)
- **Scalability:** Horizontal scaling via gunicorn workers behind Nginx reverse proxy
- **Reliability:** Graceful error handling with generic client messages + server-side logging
- **Maintainability:** Modular codebase with separation of concerns
- **Security:** Rate limiting (60 req/min), CSP headers, CORS restriction, input validation, MAX_CONTENT_LENGTH (1 MB), no exception leakage
- **Portability:** Dockerized with docker-compose (API + Nginx)

## 7. Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  CSV Data   │───▶│ Preprocessor │───▶│   Feature   │
│  (India)    │    │  (clean +    │    │  Engineer   │
│  250K rows  │    │   encode)    │    │  (31 feats) │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                               │
                      ┌────────────────────────┘
                      ▼
          ┌───────────────────────────┐
          │      Multi-Model Trainer  │
          │  ┌─────────────────────┐  │
          │  │ XGBoost    (CPU)    │  │
          │  │ LightGBM   (GPU)    │──┼──▶ GPU (NVIDIA)
          │  │ RandomForest(CPU)   │  │
          │  └─────────────────────┘  │
          └──────────┬────────────────┘
                     ▼
          ┌──────────────────────┐
          │ Multi-Modal Scoring  │
          │ (R²+MAE+MAPE)/3     │
          │ Price Range ±1.5σ   │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │   Flask + Gunicorn   │
          │  Rate limiter (60/m) │
          │  Security headers    │
          │  Input validation    │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │   Nginx Reverse      │
          │   Proxy (port 80)    │
          └──────────┬───────────┘
                     ▼
               ┌─────────────┐
               │   Client    │
               │ (Browser)   │
               └─────────────┘
```

## 8. Data Schema

| Column | Type | Description |
|---|---|---|
| State | str | Indian state (20 states) |
| City | str | City name (42 cities) |
| Property_Type | str | Apartment/Villa/Independent House |
| BHK | int | Bedrooms (1-5) |
| Size_in_SqFt | int | Built-up area |
| Price_in_Lakhs | float | Sale price in Lakhs (target) |
| Price_per_SqFt | float | Rate per sqft (key feature) |
| Year_Built | int | Construction year |
| Furnished_Status | str | Furnished/Semi-furnished/Unfurnished |
| Floor_No | int | Floor number |
| Total_Floors | int | Total building floors |
| Age_of_Property | int | Age in years |
| Nearby_Schools | int | Count nearby |
| Nearby_Hospitals | int | Count nearby |
| Public_Transport_Accessibility | str | High/Medium/Low |
| Parking_Space | binary | Yes/No |
| Security | binary | Yes/No |
| Amenities | str | Comma-separated amenities list |
| Facing | str | North/South/East/West |
| Owner_Type | str | Owner/Builder/Broker |
| Availability_Status | str | Ready_to_Move/Under_Construction |

## 9. Constraints & Assumptions

- Dataset is limited to Indian residential properties across 20 states and 42 cities
- Model does not account for market trends, interest rates, or macroeconomic factors
- Predictions are estimates, not official appraisals
- Fixed-effects per city via Price_per_SqFt rate lookup, not time-series aware

## 10. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Models | XGBoost, LightGBM (GPU), Random Forest (scikit-learn) |
| Data | numpy, pandas, scipy, joblib |
| Backend | Flask 3.x, Flask-CORS, Gunicorn |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| API Spec | OpenAPI 3.0 (api_docs.yaml) |
| Visualization | matplotlib, seaborn |
| Infrastructure | Docker, Docker Compose, Nginx |
| Testing | pytest, pytest-cov |
| Linting | ruff |
| Security | python-dotenv, CSP headers, rate limiting |

## 11. Folder Structure

```
├── .env.example              # Environment variable template
├── .gitignore
├── Dockerfile                # Gunicorn-based production image
├── PRD.md                    # Product requirements document
├── README.md                 # Project documentation
├── Security and vulnerability audit.md
├── api_docs.yaml             # OpenAPI 3.0 specification
├── docker-compose.yml        # API + Nginx services
├── gunicorn.conf.py          # Gunicorn worker config
├── nginx.conf                # Nginx reverse proxy config
├── pyproject.toml            # Project metadata + dependencies
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Pinned dependencies
│
├── data/                     # Dataset files
│   ├── house_data.csv
│   └── india_housing_prices.csv  # Main dataset (250K rows)
│
├── fig/                      # Visualization plots
├── models/                   # Trained models + artifacts
│   ├── city_rates.json       # City price-per-sqft lookup
│   ├── comparison.json       # Model comparison results
│   ├── *_*.joblib            # Serialized models
│   └── scaler.joblib         # Fitted MinMaxScaler
│
├── notebooks/                # EDA notebooks
│   └── 01_EDA.ipynb
│
├── logs/                     # Application logs
│
├── src/                      # Source code
│   ├── __init__.py           # Package init + sys.path
│   ├── api.py                # Flask REST API (routes, rate limiter, security)
│   ├── city_rates.py         # City-level price-per-sqft calculations
│   ├── compare.py            # Multi-model comparison pipeline
│   ├── data_preprocessing.py # Cleaning, encoding, splitting, scaling
│   ├── feature_engineering.py# Feature generation (31 features)
│   ├── models.py             # Model definitions (XGBoost/LightGBM/RF)
│   ├── predict.py            # Prediction logic + input validation
│   ├── scoring.py            # Multi-modal ensemble scoring
│   ├── train.py              # Single-model training pipeline
│   └── utils.py              # Logging, metrics, timestamp
│
├── static/                   # Static assets
│   └── style.css             # Application styles
│
├── templates/                # HTML templates (Jinja2)
│   ├── base.html             # Layout base
│   ├── index.html            # Landing + model comparison
│   ├── predict.html          # Prediction input form + results
│   ├── 404.html              # Not found page
│   └── 500.html              # Server error page
│
└── tests/                    # Test suite (pytest)
    ├── __init__.py
    ├── conftest.py           # Shared fixtures
    ├── test_api.py           # API endpoint tests
    ├── test_integration.py   # Integration tests
    ├── test_models.py        # Model training tests
    ├── test_predict.py       # Prediction logic tests
    ├── test_preprocessing.py # Data pipeline tests
    └── test_scoring.py       # Scoring tests
```

## 12. Future Enhancements

- Time-series features for market trends
- Deep learning models (MLP, Transformer)
- Geospatial clustering
- Interactive web dashboard (Streamlit)
- Automated retraining pipeline (CI/CD)
- Explainability (SHAP, LIME)
- Authentication layer (API keys / JWT)
