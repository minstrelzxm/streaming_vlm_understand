import cv2
import time
import threading

class CameraStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.current_frame = None
        self.running = False
        self.lock = threading.Lock()
        
        if not self.cap.isOpened():
            print(f"[CameraStream] Error: Could not open video stream at {url}")
        else:
            print(f"[CameraStream] Successfully connected to {url}")
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
            else:
                print("[CameraStream] Failed to read frame. Reconnecting...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.url)

    def get_frame(self):
        with self.lock:
            return self.current_frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

if __name__ == "__main__":
    url = input("Enter IP Camera URL (e.g., http://192.168.1.x:8080/video): ")
    cam = CameraStream(url)
    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                cv2.imshow("Test Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
