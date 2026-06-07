# House Price Prediction Full Stack ML App

This project is a full-stack machine learning application that predicts house prices using a trained Random Forest Regression model. The project includes a Python FastAPI backend and a React Vite frontend.

## Project Overview

The application allows users to enter housing details such as median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude.

The frontend sends the input data to the backend API. The backend loads the trained machine learning model and returns the predicted house price.

## Tech Stack

- Python
- Scikit-learn
- Pandas
- Joblib
- FastAPI
- React
- Vite
- TypeScript
- REST API
- Git

## Features

- Trained machine learning regression model
- FastAPI backend with prediction endpoint
- React frontend form for user input
- Live prediction result displayed in browser
- Frontend and backend connected using REST API
- CORS enabled for frontend-backend communication
- Model saved using Joblib

## Machine Learning Model

The model was trained using the California Housing dataset.

Model used:

```text
RandomForestRegressor

```

Performance:

```text
Mean Absolute Error: 32798.67
R2 Score: 0.80
```

## API Endpoints

### Home

```http
GET /
```

### Health Check

```http
GET /health
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "MedInc": 8.3,
  "HouseAge": 25,
  "AveRooms": 6.2,
  "AveBedrms": 1.1,
  "Population": 1200,
  "AveOccup": 3.2,
  "Latitude": 34.05,
  "Longitude": -118.25
}
```

Example response:

```json
{
  "predicted_price_usd": 440549.47,
  "message": "Prediction completed successfully"
}
```

## How to Run Backend

```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## How to Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on:

```text
http://localhost:5173
```

## Project Structure

```text
house-price-prediction-devops/
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── house_price_model.pkl
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── index.html
│
├── .gitignore
└── README.md
```

## Resume Highlight

Built a full-stack House Price Prediction application using Python, Scikit-learn, FastAPI, and React, with a Random Forest model achieving an R2 score of 0.80 and live predictions through a REST API.