# House Price Prediction System

**Ever wondered what a house should really cost in India?** This system uses machine learning to predict residential property values across 20 states and 42 cities, trained on 250,000+ real housing records.

Instead of just giving you one number, it combines 3 different AI models (XGBoost, LightGBM, and Random Forest) and gives you a **price range with confidence level** — so you know not just what the price might be, but how sure the system is about it.

---

## What Makes This Different

- **Not just one AI model** — It uses 3 models working together, each with different strengths
- **GPU-powered** — LightGBM runs on your NVIDIA GPU if you have one (automatically detected)
- **Confidence scores** — You get "High", "Medium", or "Low" confidence with every prediction
- **Pan-India coverage** — Works for 20 states, 42 cities, from Mumbai to Guwahati
- **Production-ready** — Not just a notebook, but a full web app with security, rate limiting, and Docker support

---

## How It Works

```
Your input (city, size, BHK, etc.)
    ↓
3 AI models analyze it
    ↓
Models vote on price (weighted by their accuracy)
    ↓
You get: predicted price + price range + confidence level
```

---

## Getting Started (5 minutes)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Models

```bash
python -m src.compare
```

This trains all 3 models and saves them. Takes about 10-15 minutes depending on your hardware.

### Step 3: Run the Web App

```bash
python -m src.api
```

Open your browser and go to `http://localhost:5000` — you'll see the prediction form.

### Step 4: Make a Prediction

Fill in the form with property details (city, size, number of bedrooms, etc.) and click "Predict". You'll get back:
- A predicted price
- A price range (what the property might realistically sell for)
- Confidence level (how sure the system is)

---

## Want to Use Docker Instead?

If you prefer Docker (recommended for production):

```bash
docker-compose up --build
```

This starts both the Flask app and Nginx reverse proxy automatically.

---

## Try the API Directly

If you prefer using the API instead of the web form:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "State": "Maharashtra",
      "City": "Mumbai",
      "Property_Type": "Apartment",
      "BHK": 3,
      "Size_in_SqFt": 1200,
      "Furnished_Status": "Semi-furnished",
      "Floor_No": 5,
      "Total_Floors": 15,
      "Age_of_Property": 10,
      "Year_Built": 2015,
      "Nearby_Schools": 5,
      "Nearby_Hospitals": 3,
      "Public_Transport_Accessibility": "High",
      "Parking_Space": "Yes",
      "Security": "Yes",
      "Facing": "East",
      "Owner_Type": "Owner",
      "Availability_Status": "Ready_to_Move"
    }
  }'
```

---

## All the API Endpoints

| What It Does | URL | Method |
|-------------|-----|--------|
| Home page | `/` | GET |
| Prediction form | `/predict` | GET |
| Make prediction | `/predict` | POST |
| Check system health | `/health` | GET |
| List available models | `/models` | GET |
| Compare model performance | `/api/comparison` | GET |
| Get confidence scores | `/api/scores` | GET |
| List all Indian states | `/api/states` | GET |
| List cities in a state | `/api/cities?state=X` | GET |
| Get city price rate | `/api/rate?state=X&city=Y` | GET |

---

## How Accurate Is It?

Pretty accurate, actually:

| Metric | What It Means | Result |
|--------|--------------|--------|
| R² Score | How well predictions match reality | **0.9962** (99.6% accurate) |
| MAPE | Average percentage error | **~5%** |
| API Response Time | How fast you get a prediction | **< 100ms** |

For context, an R² of 0.996 means the model explains 99.6% of the variation in house prices. That's excellent.

---

## What the System Checks

Before making a prediction, it validates all 18 input fields:

- **Numeric ranges**: BHK (1-10), Size (50-100,000 sqft), Floor numbers, etc.
- **Valid locations**: Must be a recognized Indian state and city
- **Logical consistency**: Floor number can't be higher than total floors
- **Required fields**: All 18 fields are required — no skipping

---

## Security Features

This isn't just a demo — it's built with security in mind:

- **Rate limiting**: 60 requests per minute per IP address (prevents abuse)
- **Security headers**: CSP, X-Frame-Options, X-XSS-Protection, etc.
- **Input validation**: All fields are checked before processing
- **No error leakage**: Errors are logged server-side, never sent to users
- **CORS control**: Only allows requests from specified origins

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Environment Variables

You can customize behavior by creating a `.env` file (copy from `.env.example`):

| Variable | Default | What It Does |
|----------|---------|--------------|
| `FLASK_DEBUG` | `false` | Enable debug mode (don't use in production) |
| `PORT` | `5000` | Which port the app runs on |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per minute per IP |
| `GUNICORN_WORKERS` | `auto` | Number of worker processes |
| `LOG_LEVEL` | `INFO` | How much detail to log |

---

## Project Structure

```
house price prediction/
├── src/                    # All the Python code
│   ├── api.py             # Web server and API routes
│   ├── predict.py         # Prediction logic
│   ├── models.py          # AI model definitions
│   ├── scoring.py         # How models work together
│   ├── data_preprocessing.py  # Data cleaning
│   └── feature_engineering.py # Creating smart features
├── tests/                  # All the tests
├── templates/              # Web pages (HTML)
├── static/                 # CSS styles
├── models/                 # Trained AI models (generated)
├── notebooks/              # Data analysis notebooks
├── fig/                    # Charts and graphs
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image definition
└── docker-compose.yml      # Docker orchestration
```

---

## Tech Stack

| Layer | What We Used |
|-------|-------------|
| AI Models | XGBoost, LightGBM (with GPU), Random Forest |
| Backend | Python, Flask, Gunicorn |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Docker, Nginx |
| Testing | pytest |

---

## For Developers

### Linting

```bash
ruff check src/ tests/
```

### Model Training

```bash
# Train a single model
python -m src.train --model xgboost

# Compare all models
python -m src.compare
```

### API Documentation

Full OpenAPI 3.0 spec is in `api_docs.yaml`.

---

## License

MIT — use it however you want.

---

**Built with care at IIT Guwahati**
