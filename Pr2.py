import pandas as pd
import numpy as np

# Створюємо великий DataFrame для тестування
data = {
    'text_length': np.random.randint(50, 1000, size=10**6),  # Довжина тексту
    'word_count': np.random.randint(10, 200, size=10**6),  # Кількість слів
    'sentiment_score': np.random.uniform(-1, 1, size=10**6)  # Тональність тексту
}
df = pd.DataFrame(data)

# Визначаємо кастомну метрику: середня довжина слова, помножена на тональність

def custom_metric(row):
    avg_word_length = row['text_length'] / row['word_count'] if row['word_count'] > 0 else 0
    return avg_word_length * row['sentiment_score']

# Використовуємо .apply() для розрахунку метрики
df['custom_metric'] = df.apply(custom_metric, axis=1)

# Виводимо перші 5 рядків
print(df.head())

