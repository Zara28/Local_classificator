from TaskClassifier import TaskClassifier
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("test_dataset.csv")

    classifier = TaskClassifier()
    classifier.train(texts=df['text'].tolist(), labels=df['category'].tolist())

    new_requests = [
        "Нужен человек для отрисовки графики на канал",
        "Требуется переписать базу с MS SQL на постгрес и настроить бэкапы",
        "Ищем специалиста для анализа метрик оттока из приложения",
        "Надо запустить рекламу в телеге на широкую аудиторию"
    ]

    for req in new_requests:
        result = classifier.predict(req)
        print(f"Входящее ТЗ: «{result['text']}»")
        print(f"Категория:   {result['category']}")
        print(f"Уверенность: {result['confidence']}%\n")