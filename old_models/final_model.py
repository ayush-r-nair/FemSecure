import cv2
from ultralytics import YOLO
import time

model = YOLO("yolov8x.pt")

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net_final.caffemodel"
genderNet = cv2.dnn.readNet(genderModel, genderProto)

mean_value = [78.4263377603, 87.7689143744, 114.895847746]
gender_list = ["Female", "Male"]

faceProto = "deploy.prototxt"
faceModel = "res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

source = r"Input Videos\vid10.mp4"
vid_capture = cv2.VideoCapture(source)

fps = vid_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output1.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

class_names = model.names
person_class_index = None
for cid, cname in class_names.items():
    if cname == "person":
        person_class_index = cid
        break

def detect_face(frame, x1, y1, x2, y2):
    person_roi = frame[y1:y2, x1:x2]
    if person_roi.size == 0:
        return None

    h, w = person_roi.shape[:2]
    blob = cv2.dnn.blobFromImage(
        person_roi, 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            fx1, fy1, fx2, fy2 = box.astype("int")
            face = person_roi[fy1:fy2, fx1:fx2]
            return face

    return None

frame_count = 0
last_displayed_time = -1

while True:
    ret, frame = vid_capture.read()
    if not ret:
        break

    start_time = time.time()
    male_count = 0
    female_count = 0

    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        cls = int(box.cls)
        if cls != person_class_index:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        face = detect_face(frame, x1, y1, x2, y2)
        if face is None:
            continue

        if face.shape[0] < 40 or face.shape[1] < 40:
            continue

        face_blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), mean_value, swapRB=False
        )
        genderNet.setInput(face_blob)
        gender_preds = genderNet.forward()
        gender = gender_list[gender_preds[0].argmax()]

        if gender == "Male":
            male_count += 1
        else:
            female_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 129), 2)
        cv2.putText(
            frame,
            gender,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    text_time = int(frame_count / fps)
    if text_time % 5 == 0 and text_time != last_displayed_time:
        last_displayed_time = text_time
        cv2.putText(
            frame,
            f"Males: {male_count}",
            (10, frame_height - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Females: {female_count}",
            (frame_width - 180, frame_height - 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (255, 0, 0),
            2,
        )

    out.write(frame)
    cv2.imshow("Person and Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    elapsed_time = time.time() - start_time
    time.sleep(max(1.0 / fps - elapsed_time, 0))
    frame_count += 1

vid_capture.release()
out.release()
cv2.destroyAllWindows()
