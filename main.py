import cv2
from gender_analytics import process_gender
from sos_detection import process_sos

source = 0 
cap = cv2.VideoCapture(source)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "femsecure_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps if fps > 0 else 25,
    (w, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, male_count, female_count = process_gender(frame)
    sos_active = process_sos(frame)

    if sos_active:
        cv2.putText(
            frame,
            "!!! EMERGENCY ALERT !!!",
            (w // 4, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            4
        )

    cv2.putText(
        frame,
        f"Males: {male_count} | Females: {female_count}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)
    cv2.imshow("FemSecure", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
