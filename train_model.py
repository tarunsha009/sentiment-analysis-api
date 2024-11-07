import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessor import TextPreprocessor


def train_text_classifier(texts, labels):

    preprocessor = TextPreprocessor()

    cleaned_texts = preprocessor.preprocess_text(texts)

    x_train, x_test, y_train, y_test = train_test_split(cleaned_texts, labels, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', preprocessor.vectorizer),
        ('classifier', LogisticRegression())
    ])

    pipeline.fit(x_train, y_train)

    prediction = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    print("Model Accuracy:", accuracy)

    model_filename = "text_classifier_model.joblib"
    joblib.dump(pipeline, model_filename)

    print(f"Model saved to {model_filename}")

    return pipeline, accuracy


if __name__ == "__main__":
    # Positive Sentiment
    texts = [
        "I absolutely love this product, it works perfectly!",
        "Fantastic experience! Highly recommend.",
        "This is amazing, couldn't be happier.",
        "Great quality and fantastic service.",
        "I'm very pleased with my purchase.",
        "Exceeded my expectations, really satisfied!",
        "Couldn't be better, absolutely fantastic!",

        # Neutral Sentiment
        "It was okay, nothing special.",
        "The product is average, as expected.",
        "Not bad, but nothing amazing either.",
        "I don't have any strong feelings about this.",
        "It's alright, just as I thought.",
        "Meh, it's neither good nor bad.",
        "The quality is fine, nothing more.",

        # Negative Sentiment
        "I'm really disappointed with this product.",
        "Terrible experience, would not recommend.",
        "Not great at all, poor quality.",
        "The product broke after a few uses, awful!",
        "Definitely not worth the money.",
        "I don't like it, very unhappy.",
        "The quality isn't great, very poor experience.",
        "It didn't meet my expectations at all."
    ]

    labels = [
        2, 2, 2, 2, 2, 2, 2,  # Positive labels
        1, 1, 1, 1, 1, 1, 1,  # Neutral labels
        0, 0, 0, 0, 0, 0, 0, 0  # Negative labels
    ]

    # Train the model
    model, accuracy = train_text_classifier(texts, labels)
    print("Sentiment Analysis Model Accuracy:", accuracy)

