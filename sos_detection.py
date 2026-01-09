import cv2
import mediapipe as mp
import time
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

gesture_buffer = deque(maxlen=15)
last_sos_time = 0
COOLDOWN = 10

sos_active = False
sos_display_until = 0
SOS_DISPLAY_DURATION = 3


def is_peace_sign(landmarks):
    def is_extended(tip, pip):
        return tip.y < pip.y

    index_ext = is_extended(
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    )
    middle_ext = is_extended(
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    )
    ring_folded = (
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y >
        landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_folded = (
        landmarks[mp_hands.HandLandmark.PINKY_TIP].y >
        landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    )

    return index_ext and middle_ext and ring_folded and pinky_folded


def process_sos(frame):
    global sos_active, sos_display_until, last_sos_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    detected = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if is_peace_sign(hand_landmarks.landmark):
                detected = 1
                cv2.putText(
                    frame,
                    "Peace Sign Detected",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

    gesture_buffer.append(detected)
    now = time.time()

    if sum(gesture_buffer) >= 6 and (now - last_sos_time) > COOLDOWN:
        sos_active = True
        sos_display_until = now + SOS_DISPLAY_DURATION
        last_sos_time = now
        gesture_buffer.clear()

    if sos_active:
        cv2.putText(
            frame,
            "SOS CONFIRMED",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            4
        )

        if time.time() > sos_display_until:
            sos_active = False

    return sos_active