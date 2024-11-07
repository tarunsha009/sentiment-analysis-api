import joblib
from flask import Flask, request
from flask_restx import Api, fields, Resource

app = Flask(__name__)
api = Api(app, version='1.0', title='Text Classifier API',
          description='A simple API for text classification using a trained model')

model_filename = "text_classifier_model.joblib"
model = joblib.load(model_filename)

print(f"Model loaded from {model_filename}")

text_input = api.model('TextInput', {
    'text': fields.String(required=True, description='Text to classify')
})


@api.route('/predict')
class Predict(Resource):

    @api.expect(text_input)
    def post(self):

        data = request.get_json()

        if 'text' not in data:
            return {'error': 'No text provided'}, 400

        text = data['text']

        prediction = model.predict([text])[0]
        confidence = max(model.predict_proba([text])[0])

        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_label = sentiment_map[prediction]

        return {
            "text": text,
            "predicted_sentiment": sentiment_label,
            "confidence": round(confidence, 2)
        }


if __name__ == '__main__':
    app.run(debug=True)


