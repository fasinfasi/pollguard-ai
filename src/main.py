import cv2
from detection.person_detector import PersonDetector

def main():
    video_path = r"C:\Users\HP\Repositories\Projects\PollGuard\data\kids_playing.mp4"
    output_path = r"C:\Users\HP\Repositories\Projects\PollGuard\outputs\videos"

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    detector = PersonDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        for det in detections:
            x1,y1,x2,y2 = det['bbox']

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        out.write(frame)
        cv2.namedWindow("Processed frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Processed frame", frame)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()