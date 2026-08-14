import pandas as pd
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer


class TaskClassifier:
    def __init__(self, model_name: str = 'cointegrated/rubert-tiny2'):
        """
        Инициализация классификатора.
        Загружает легковесную языковую модель для получения семантических эмбеддингов.
        """
        print(f"Загрузка языковой модели {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        self.clf = LogisticRegression(random_state=42, C=10.0, max_iter=1000)
        self.is_trained = False

    def train(self, texts: list, labels: list):
        """
        Обучение классификатора на размеченных текстах.
        """
        print("Векторизация обучающих данных (может занять пару секунд)...")
        x_train = self.embedder.encode(texts)

        print("Обучение логистической регрессии...")
        self.clf.fit(x_train, labels)
        self.is_trained = True
        print("Модель успешно обучена!\n")

    def predict(self, text: str) -> dict:
        """
        Классификация нового текста.
        Возвращает словарь с предсказанной категорией и уровнем уверенности модели.
        """
        if not self.is_trained:
            raise ValueError("Модель еще не обучена. Сначала вызовите метод train().")

        # Переводим входящий текст в семантический вектор
        vector = self.embedder.encode([text])

        # Получаем предсказание и вероятности
        prediction = self.clf.predict(vector)[0]
        probabilities = self.clf.predict_proba(vector)[0]
        max_prob = max(probabilities) * 100

        return {
            "text": text,
            "category": prediction,
            "confidence": round(max_prob, 1)
        }