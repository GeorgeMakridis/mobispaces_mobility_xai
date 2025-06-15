# Mobispaces: Demo #1 T5.1
## AI Model for Ship Position Prediction

### About
This Docker container hosts a Flask REST web application designed for predicting future positions of ships using a pre-trained XGBoost model. The model leverages historical data such as longitude, latitude, speed, bearing, and time components to forecast ship locations accurately. This application is crucial for enhancing maritime navigation and safety by providing:
- Predictive analytics on ship movements based on historical and real-time data.
- API endpoints for fetching real-time position forecasts.
- A scalable solution that integrates seamlessly with maritime operational systems.

The application uses a dataset comprising ship coordinates, speeds, bearings, and other navigational parameters to ensure accurate and reliable predictions.

### Deployment

Follow these steps to deploy the ship position prediction model as a Docker container on your local machine:

1. **Prerequisites**: Ensure Docker is installed on your system.
2. **Setup**: Clone or download the project files to your local machine.
3. **Build Container**:
   Navigate to the project directory and build the Docker image:
   ```bash
   cd PredictionModel
   docker build -t ship-position-prediction .

