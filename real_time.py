import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('saved_model')

class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']  # Add all your class names here


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    img_array = image.img_to_array(resized)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = preprocess_frame(frame)

    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    emotion_label = class_labels[predicted_class]

    cv2.putText(frame, f'Emotion: {emotion_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()