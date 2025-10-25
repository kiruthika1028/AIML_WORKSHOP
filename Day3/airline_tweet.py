import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load
df = pd.read_csv(r"C:\Users\kiruthika\Downloads\aiml-workshop\Day3\Tweets.csv")


# Basic cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

df['clean_text'] = df['text'].apply(clean_text)

# Vectorize (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['clean_text']).toarray()  # toarray() ok if fits memory

# Encode labels
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(df['airline_sentiment'])
y = to_categorical(y_int)

# Train-test split with stratify to preserve class distribution
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, df['clean_text'], test_size=0.2, random_state=42, stratify=y_int)

# Build model
model = Sequential([
    Dense(256, input_dim=X.shape[1], activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=20, batch_size=64,
          validation_data=(X_test, y_test), callbacks=[es])

# Evaluate and detailed metrics
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

preds_proba = model.predict(X_test)
preds = np.argmax(preds_proba, axis=1)
true = np.argmax(y_test, axis=1)

print("\nClassification report:")
print(classification_report(true, preds, target_names=label_encoder.classes_))

print("\nConfusion matrix:")
print(confusion_matrix(true, preds))
