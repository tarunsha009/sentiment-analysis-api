import re

from sklearn.feature_extraction.text import TfidfVectorizer


class TextPreprocessor:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def preprocess_text(self, texts):
        return [self.clean_text(text) for text in texts]


    def vectorize_text(self, texts):

        return self.vectorizer.fit_transform(texts)
