import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Resize the image to 28x28 using bilinear interpolation
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to the range [0, 1]
    normalized_image = gray_image.astype('float32') / 255.0
    print(normalized_image)

    return normalized_image


image_path = 'sign_ps.jpg'
processed_image = preprocess_image(image_path)

# Display the processed image using OpenCV or save it
# cv2.imshow('Processed Image', processed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Convert the processed image to a NumPy array of dimensions 28x28
numpy_array = np.array(processed_image)
print(numpy_array.shape)  # Output: (28, 28)
print(numpy_array)

checkpoint_path = "training/cp_2.ckpt"
def create_model():

  ### START CODE HERE

  # Define the model
  # Use no more than 2 Conv2D and 2 MaxPooling2D
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(25, activation='softmax')
  ])


  model.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics=['accuracy'])
  return model

model = create_model()

model.load_weights(checkpoint_path)

# model.summary()
image_to_predict = np.expand_dims(numpy_array, axis=(0, -1))
print(image_to_predict.shape)
prediction = model.predict(image_to_predict)

abc = "abcdefghijklmnopqrstuvwxyz"
prediction = prediction[0].tolist()
print("abc - ", len(abc))
print("pred - ", len(prediction))
for letter, prob in zip(abc, prediction):
   print(letter, " - ", prob)

# Find the maximum element in the list
max_element = max(prediction)
max_index = prediction.index(max_element)

print(abc[max_index])