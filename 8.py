import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

vocab_size, max_length = 10000, 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print("Training Samples:", len(x_train))
print("Testing Samples:", len(x_test))

x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", accuracy)
print("Test Loss:", loss)

word_index = imdb.get_word_index()

def encode_review(review):
    return [word_index.get(word, 2) + 3 for word in review.lower().split()]

review = input("\nEnter Movie Review: ")
encoded_review = encode_review(review)
padded_review = pad_sequences([encoded_review], maxlen=max_length)

prediction = model.predict(padded_review)
print("\nPrediction Score:", prediction[0][0])

if prediction[0][0] >= 0.5:
    print("Sentiment: Positive Review")
else:
    print("Sentiment: Negative Review")