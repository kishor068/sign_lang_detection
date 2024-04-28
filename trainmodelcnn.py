from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import TensorBoard

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

# Assuming each frame is a 2D array (e.g., grayscale image)
# You might need to adjust the input shape based on your actual data format
input_shape = (sequence_length, 63, 1) # Example: sequence_length frames, 63x63 pixels, 1 channel (grayscale)

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Use a different loss function
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')


import numpy as np
from sklearn.metrics import accuracy_score

# Assuming y_pred is the output from model.predict(X_test)
y_pred = model.predict(X_test)
# Convert probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)

# Convert one-hot encoded y_test to class labels
y_test_labels = np.argmax(y_test, axis=1)

# Now you can calculate accuracy
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print('Accuracy:', accuracy)
