import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from test_data import raw_data

df = pd.DataFrame(raw_data, columns=['text', 'category'])

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )),
    ('classifier', LogisticRegression(random_state=42, C=1.0))
])

print("Обучение модели...")
pipeline.fit(df['text'], df['category'])
print("Обучение завершено\n")

predictions = pipeline.predict(df['text'])
print("Отчет по качеству классификации:")
print(classification_report(df['category'], predictions))

test_requests = [
    "Нужно написать SQL скрипт для расчета конверсии по дням",
    "Ищу человека, который нарисует красивые иконки",
    "Требуется бэкенд на python и база данных",
    "Нужен спец по закупке рекламы в телеграм-каналах",
    "Срочно починить вылеты на айфоне"
]

print("-" * 40)
print("Тестирование на новых заявках:\n")
for req in test_requests:
    predicted_category = pipeline.predict([req])[0]

    probabilities = pipeline.predict_proba([req])[0]
    max_prob = max(probabilities) * 100

    print(f"ТЗ: «{req}»")
    print(f"Категория: {predicted_category} (уверенность: {max_prob:.1f}%)\n")