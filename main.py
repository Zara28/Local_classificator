import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("test_dataset.csv")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )),
    ('classifier', LogisticRegression(random_state=42, C=1.0))
])

df = df.sample(frac=1, random_state=42)

df_test = df.iloc[:len(df) // 2]
df_validation = df.iloc[len(df) // 2:]

print("Обучение модели...")
pipeline.fit(df_test['text'], df_test['category'])
print("Обучение завершено\n")

predictions = pipeline.predict(df_test['text'])
print("Отчет по качеству классификации:")
print(classification_report(df_test['category'], predictions))

print("-" * 40)
print("Тестирование на новых заявках:\n")
texts = df_validation['text']
predictions = pipeline.predict(texts)
probabilities = pipeline.predict_proba(texts)

# 2. Итерируемся по текстам и уже готовым результатам с помощью zip
for req, pred, probs in zip(texts, predictions, probabilities):
    max_prob = max(probs) * 100

    print(f"ТЗ: «{req}»")
    print(f"Категория: {pred} (уверенность: {max_prob:.1f}%)\n")