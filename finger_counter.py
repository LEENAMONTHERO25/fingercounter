import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open the camera
cap = cv2.VideoCapture(0)

def count_fingers(landmarks):
    """
    Count the number of fingers raised based on landmarks.
    """
    fingers = []
    # Thumb: Compare x-coordinates of tip and joint
    fingers.append(landmarks[4][0] > landmarks[3][0])
    # Other fingers: Compare y-coordinates of tip and lower joint
    for i in range(1, 5):
        fingers.append(landmarks[4 * i + 4][1] < landmarks[4 * i + 2][1])
    return fingers.count(True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the image
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks and connections on the hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            h, w, _ = frame.shape
            landmarks = [(int(x * w), int(y * h)) for x, y in landmarks]

            # Count fingers
            fingers = count_fingers(landmarks)
            cv2.putText(frame, f'Fingers: {fingers}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Finger Counter", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
