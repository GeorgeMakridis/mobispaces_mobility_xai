# MobiSpaces Mobility XAI

A comprehensive eXplainable AI (XAI) system for maritime ship trajectory prediction, consisting of three microservices that provide end-to-end solutions for predicting and explaining ship movements.

## Architecture

The system is built as a microservices architecture with three main components:

### 1. mobiAI (AI Model Service)
- **Purpose**: Core AI prediction engine for ship trajectory forecasting
- **Technology**: Flask REST API with XGBoost models
- **Port**: 8880
- **Features**:
  - Spider chart generation for platform performance comparison
  - Trip data management and trajectory data processing
  - Map data processing with GeoJSON format support
  - Two prediction models: Short-term (20m horizon) and Mid-term (30m horizon)

### 2. mobiXAI (Explainable AI Service)
- **Purpose**: Provides AI model interpretability and explanations
- **Technology**: Flask REST API with LIME (Local Interpretable Model-agnostic Explanations)
- **Port**: 8881
- **Features**:
  - Feature importance analysis
  - LIME explanations for individual predictions
  - GPT-4 integration for human-readable explanations
  - Instance-specific explanations

### 3. mobiUI (User Interface)
- **Purpose**: Web-based dashboard for visualizing predictions and explanations
- **Technology**: Streamlit with PyDeck for interactive maps
- **Port**: 8501
- **Features**:
  - Interactive map visualization
  - Model selection between prediction horizons
  - Real-time explanations display
  - Performance benchmarking

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ (for local development)

### Running the Services

1. **Start mobiAI service:**
   ```bash
   cd mobiAI
   docker-compose up -d
   ```

2. **Start mobiXAI service:**
   ```bash
   cd mobiXAI
   docker-compose up -d
   ```

3. **Start mobiUI service:**
   ```bash
   cd mobiUI
   docker-compose up -d
   ```

### Accessing the Application

- **UI Dashboard**: http://localhost:8501
- **AI API**: http://localhost:8880
- **XAI API**: http://localhost:8881

## Features

- **Multi-horizon Prediction**: Supports both 20-minute and 30-minute prediction horizons
- **Real-time Processing**: API-based architecture enables real-time predictions
- **Explainable AI**: Provides transparency into model decisions
- **Interactive Visualization**: User-friendly interface for exploring predictions
- **Performance Benchmarking**: Compares model performance across different hardware platforms

## Technical Stack

### Backend Technologies
- Python 3.8-3.9
- Flask (REST API framework)
- Streamlit (Web framework)
- XGBoost (Machine learning model)
- LIME (Explainable AI library)
- OpenAI GPT-4 (Natural language explanation generation)

### Data Processing
- Pandas (Data manipulation and analysis)
- NumPy (Numerical computations)
- GeoJSON (Geographic data format)
- Matplotlib (Chart generation)

### Visualization
- PyDeck (Interactive map rendering)
- Streamlit (Web interface components)
- Matplotlib (Static chart generation)

### Deployment
- Docker (Containerization)
- Gunicorn (WSGI server)
- Microservices Architecture

## Use Cases

This system is designed for **maritime navigation and safety**, specifically:
- Ship trajectory prediction
- Maritime safety enhancement through predictive analytics
- Operational intelligence for maritime operations
- Explainable AI for transparent predictions

## Recent Fixes

### Index Mismatch Fix
- **Issue**: Map tooltips showed original dataset indexes while explanation system expected sequential indexes
- **Solution**: Modified `mobiAI/app.py` to use reset DataFrame indexes instead of original dataset indexes
- **Result**: Map tooltips now show sequential indexes (0,1,2,3...) that work correctly with the explanation system

## Disclaimer

**This software has been developed thanks to the MobiSpaces project (Grant agreement ID: 101070279)**

---

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here] 