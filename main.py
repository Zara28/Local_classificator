import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

print("Загрузка модели rubert-tiny2...")
embedder = SentenceTransformer('cointegrated/rubert-tiny2')

# 1. Загружаем датасет
df = pd.read_csv("test_dataset.csv")

df_train, df_test = train_test_split(
    df,
    test_size=0.5,
    random_state=42,
    stratify=df['category']
)

train_labels = df_train['category'].tolist()
test_labels = df_test['category'].tolist()

print("Векторизация обучающих данных...")
X_train = embedder.encode(df_train['text'].tolist())

print("Векторизация тестовых данных...")
X_test = embedder.encode(df_test['text'].tolist())

print("Обучение классификатора...\n")
clf = LogisticRegression(random_state=42, C=10.0, max_iter=1000)
clf.fit(X_train, train_labels)

predictions = clf.predict(X_test)
print("Отчет по качеству:")
print(classification_report(test_labels, predictions))
