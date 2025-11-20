import cv2
from deepface import DeepFace
import pyttsx3
import threading

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speech speed

# Function to speak in a separate thread
def speak(emotion):
    engine.say(f"You seem {emotion}")
    engine.runAndWait()

# Load video from webcam
cap = cv2.VideoCapture(0)
last_emotion = ""

window_name = "Emotion Recognition"
cv2.namedWindow(window_name)

while True:
    key, img = cap.read()
    if not key:
        break

    # Analyze emotion
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    emotion = results[0]['dominant_emotion']

    # Display emotion
    cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow(window_name, img)

    # Speak emotion if it changes
    if emotion != last_emotion:
        threading.Thread(target=speak, args=(emotion,), daemon=True).start()
        last_emotion = emotion

    # Detect 'q' or window close
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    # Check if window was closed manually
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
