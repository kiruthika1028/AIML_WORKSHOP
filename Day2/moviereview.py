import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
except Exception:
    from keras.models import Sequential
    from keras.layers import Dense

data ={
    'review': [
        'I loved the movie, it was fantastic!',
        'What a great film, I will watch it again.',
        'An amazing experience, truly a masterpiece.',
        'I hated the movie, it was terrible.',
        'What a boring film, I will not watch it again.',
        'A dreadful experience, truly a disaster.'
    ],
    'sentiment': [
        'positive',
        'positive',
        'positive',
        'negative',
        'negative',
        'negative'
    ]
}
df = pd.DataFrame(data)
vectoroizer = CountVectorizer()
X = vectoroizer.fit_transform(data['review']).toarray()
le = LabelEncoder() 
y = le.fit_transform(data['sentiment'])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(8, input_dim=X.shape[1], activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=2, verbose=0)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy*100:.2f}%')

sample_reviews = [
    'I really enjoyed the movie, it was wonderful!']
sample_X = vectoroizer.transform(sample_reviews).toarray()
predictions = model.predict(sample_X)

print("\n--- Sample Review Prediction ---")
print("Predicted Sentiment:", 'positive' if predictions[0][0] > 0.5 else 'negative')