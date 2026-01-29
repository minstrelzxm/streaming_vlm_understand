import cv2
import time
import threading
from model_loader import VLMHandler
from camera_stream import CameraStream
from PIL import Image

def main():
    print("Initializing Real-time VLM Flow...")
    
    # Initialize Model (this takes time)
    try:
        vlm = VLMHandler()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Get Camera URL
    url = input("Enter IP Camera URL (e.g., http://192.168.1.x:8080/video): ")
    if not url:
        print("No URL provided. Exiting.")
        return

    # Initialize Camera
    camera = CameraStream(url)
    
    # Wait for camera to start
    time.sleep(2)
    if not camera.running:
        print("Camera failed to connect.")
        return

    print("Starting loop. Press 'q' to quit.")

    last_analysis_time = 0
    analysis_interval = 5.0 # seconds
    current_description = "Waiting for analysis..."
    
    def analyze_task(frame_rgb):
        nonlocal current_description
        try:
            pil_image = Image.fromarray(frame_rgb)
            desc = vlm.analyze_frame(pil_image)
            current_description = desc
            print(f"\n[Analysis]: {desc}")
        except Exception as e:
            print(f"Analysis error: {e}")

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                current_description = "No signal..."
                time.sleep(0.1)
                continue

            # Check if it's time to analyze
            now = time.time()
            if now - last_analysis_time >= analysis_interval:
                # Launch analysis in a separate thread to not block UI
                # Need to convert to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                threading.Thread(target=analyze_task, args=(frame_rgb,), daemon=True).start()
                last_analysis_time = now

            # Overlay Text
            # Wrap text to fit screen
            display_frame = frame.copy()
            
            # Simple text overlay logic
            font = cv2.FONT_HERSHEY_SIMPLEX
            y0, dy = 30, 25
            for i, line in enumerate(current_description.split('\n')):
                # simple word wrap simulation or just truncation
                # For now just print the raw text, maybe truncated
                y = y0 + i*dy
                cv2.putText(display_frame, line[:80], (10, y), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Real-time VLM", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
