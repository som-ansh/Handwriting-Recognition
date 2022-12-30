import tensorflow as tf 

 

# Load the MNIST dataset 

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data() 

 

# Preprocess the data by reshaping it and scaling it to a range of 0-1 

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)) 

X_train = X_train.astype("float32") / 255 

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)) 

X_test = X_test.astype("float32") / 255 

 

# Convert the labels to one-hot encoded arrays 

y_train = tf.keras.utils.to_categorical(y_train) 

y_test = tf.keras.utils.to_categorical(y_test) 

 

# Define the model architecture 

model = tf.keras.models.Sequential() 

model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))) 

model.add(tf.keras.layers.MaxPooling2D((2, 2))) 

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu")) 

model.add(tf.keras.layers.MaxPooling2D((2, 2))) 

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu")) 

model.add(tf.keras.layers.Flatten()) 

model.add(tf.keras.layers.Dense(64, activation="relu")) 

model.add(tf.keras.layers.Dense(10, activation="softmax")) 

 

# Compile the model 

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) 

 

# Train the model 

model.fit(X_train, y_train, epochs=5, batch_size=64) 

 

# Evaluate the model on the test data 

test_loss, test_acc = model.evaluate(X_test, y_test) 

print("Test loss:", test_loss) 

print("Test accuracy:", test_acc) 
