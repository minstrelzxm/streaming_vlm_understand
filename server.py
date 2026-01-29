import cv2
import base64
import numpy as np
import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model_loader import VLMHandler
from PIL import Image
import io
import json

# Initialize VLM (Global)
print("Initializing VLM Model...")
vlm_handler = VLMHandler()
print("VLM Model Ready.")

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        frame_buffer = []
        
        while True:
            # Receive data (JSON)
            data_str = await websocket.receive_text()
            try:
                payload = json.loads(data_str)
                mode = payload.get("mode", "image")
                encoded_data = payload.get("data")
            except json.JSONDecodeError:
                continue

            if not encoded_data:
                continue

            # Decode Image
            if "," in encoded_data:
                header, encoded = encoded_data.split(",", 1)
            else:
                encoded = encoded_data
            
            try:
                image_data = base64.b64decode(encoded)
                # Convert to numpy for OpenCV (VideoWriter) or PIL for VLM
                nparr = np.frombuffer(image_data, np.uint8)
                img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Image decode error: {e}")
                continue

            # LOGIC SPLIT
            if mode == "image":
                # Clear buffer if switching modes
                frame_buffer = [] 
                
                # Run Inference
                loop = asyncio.get_event_loop()
                description = await loop.run_in_executor(None, vlm_handler.analyze_frame, img_pil)
                print(f"Image Analysis: {description[:50]}...")
                await websocket.send_text(description)

            elif mode == "video":
                # Add to buffer
                frame_buffer.append(img_cv2)
                
                # Check condition to trigger analysis (e.g. 15 frames ~ 3 seconds at 5fps)
                if len(frame_buffer) >= 15:
                    print(f"Processing video batch ({len(frame_buffer)} frames)...")
                    
                    # Create Temp Video
                    height, width, layers = img_cv2.shape
                    import tempfile
                    import os
                    
                    temp_video_path = "temp_stream.mp4"
                    # MP4V codec
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(temp_video_path, fourcc, 5, (width, height))

                    for frame in frame_buffer:
                        video.write(frame)
                    
                    video.release()
                    
                    # Analyze
                    loop = asyncio.get_event_loop()
                    description = await loop.run_in_executor(None, vlm_handler.analyze_video, temp_video_path)
                    
                    print(f"Video Analysis: {description[:50]}...")
                    await websocket.send_text(description)
                    
                    # Clear buffer to start next chunk
                    frame_buffer = []
            
            # Small sleep to yield
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        print("Client disconnected")
if __name__ == "__main__":
    # Get local IP
    import socket
    hostname = socket.gethostname()
    try:
        # connect to an external server (doesn't have to be reachable) to get the local interface IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    print(f"\n\n{'='*50}")
    print(f"Server starting! Connect your iPhone to:")
    print(f"👉 https://{local_ip}:8000")
    print(f"{'='*50}\n\n")

    # Run with SSL
    uvicorn.run(
        "server:app", 
        host="0.0.0.0", 
        port=8000, 
        ssl_keyfile="key.pem", 
        ssl_certfile="cert.pem"
    )
