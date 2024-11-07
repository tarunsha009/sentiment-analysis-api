# Sentiment Analysis API

This project is a **Sentiment Analysis API** built with Flask and Flask-RESTx. It classifies input text into **Positive**, **Neutral**, or **Negative** sentiment categories, using a trained machine learning model.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model and API Details](#model-and-api-details)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The API is designed to classify text-based inputs by sentiment, helping users analyze the tone of sentences or reviews. It leverages a Logistic Regression model with TF-IDF vectorization for efficient text classification.

---

## Features

- **Sentiment Classification**: Predicts Positive, Neutral, or Negative sentiment for input text.
- **Confidence Score**: Returns the model's confidence level for each prediction.
- **REST API**: Built with Flask and Flask-RESTx, with automatic API documentation.
- **Error Handling**: Provides clear error messages for invalid inputs.

---

## Installation

### 1. Clone the Repository
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-api.git
   cd sentiment-analysis-api
   ```

### 2. Create a Virtual Environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

### 3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

### 4. Train the Model
   Run `train_model.py` to preprocess data, train the model, and save it as `text_classifier_model.joblib`.
   ```bash
   python train_model.py
   ```

### 5. Run the API
   Start the Flask API server.
   ```bash
   python app.py
   ```

---

## Usage

To classify the sentiment of a sentence, send a POST request to `/predict` with the text you want to analyze. 

**Example Request**:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "The product is amazing!"}'
```

**Example Response**:
```json
{
  "text": "The product is amazing!",
  "predicted_sentiment": "Positive",
  "confidence": 0.82
}
```

---

## Project Structure

```
sentiment-analysis-api/
├── app.py              # Flask API with Flask-RESTx
├── preprocessor.py     # Preprocessing module for text data
├── train_model.py      # Model training script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## Model and API Details

- **Model**: Logistic Regression with TF-IDF vectorization (bigrams)
- **Training Data**: Custom dataset with labeled Positive, Neutral, and Negative examples
- **Confidence Threshold**: Optional threshold set to avoid low-confidence predictions (configurable in `app.py`)

---

## Future Improvements

- **Expand Training Data**: Incorporate larger datasets for improved accuracy.
- **Advanced Models**: Experiment with transformer-based models for nuanced sentiment analysis.
- **Deployment**: Deploy the API on cloud services like AWS, GCP, or Azure.

---

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues to improve the project.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
