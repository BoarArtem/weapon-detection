from ultralytics import YOLO
import cv2

def inference_photo_model(img_path, model_path):
    model = YOLO(model_path)
    results = model(img_path)

    annotated_frame = results[0].plot()

    cv2.imshow("Result", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def realtime_inference(model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        result = model(frame)

        annotated_frame = result[0].plot()

        cv2.imshow("Weapon detect", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



# inference_photo_model("img/5.webp", "runs/detect/train/weights/best.pt")
# realtime_inference("runs/detect/train/weights/best.pt")