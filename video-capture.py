import cv2
import threading
import queue
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sounddevice as sd
import soundfile as sf
import logging
from datetime import datetime

class BasicDetectionSystem:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize queues with smaller sizes
        self.frame_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=5)
        
        # Initialize YOLO with device specification
        try:
            self.model = YOLO('models/yolov8n.pt')
            self.device = 'cpu'  # Change to 'cuda' if using GPU
            self.logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
        
        # State management
        self.running = True
        
        # Camera settings - reduced resolution for better performance
        self.frame_width = 416  # Standard YOLO input size
        self.frame_height = 416
        self.fps = 30
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"recordings_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir}")
        
    def capture_video(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(self.output_dir / 'output.mp4'),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        frame_count = 0
        try:
            while self.running:
                ret, frame = cap.read()
                if ret:
                    # Resize frame for consistency
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    out.write(frame)
                    
                    # Only process every 2nd frame
                    if frame_count % 2 == 0 and not self.frame_queue.full():
                        self.frame_queue.put(frame.copy())
                    frame_count += 1
                else:
                    self.logger.error("Failed to capture frame")
                    break
                    
        finally:
            cap.release()
            out.release()
            self.logger.info("Video capture stopped")

    def process_frames(self):
        while self.running:
            if not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                    if frame is None:
                        continue
                        
                    # Run inference
                    results = self.model(frame, device=self.device, conf=0.25)
                    
                    # Draw results on a copy of the frame
                    annotated_frame = frame.copy()
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            # Get box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Get class name and confidence
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = self.model.names[cls]
                            
                            # Draw box and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{name} {conf:.2f}"
                            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )
                    
                    # Put annotated frame in display queue
                    if not self.display_queue.full():
                        self.display_queue.put(annotated_frame)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Frame processing error: {str(e)}")
                    continue

    def display_frames(self):
        while self.running:
            if not self.display_queue.empty():
                try:
                    frame = self.display_queue.get_nowait()
                    if frame is not None:
                        cv2.imshow('Object Detection', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Display error: {str(e)}")
                    continue
            time.sleep(0.01)  # Small delay to prevent high CPU usage

    def record_audio(self):
        try:
            with sf.SoundFile(
                str(self.output_dir / 'audio.wav'),
                mode='w',
                samplerate=44100,
                channels=1
            ) as audio_file:
                
                def audio_callback(indata, frames, time, status):
                    if status:
                        self.logger.warning(f"Audio status: {status}")
                    audio_file.write(indata)
                
                with sd.InputStream(
                    samplerate=44100,
                    channels=1,
                    callback=audio_callback,
                    blocksize=2048  # Increased block size for better performance
                ):
                    while self.running:
                        time.sleep(0.1)
                        
        except Exception as e:
            self.logger.error(f"Audio recording error: {e}")

    def run(self):
        try:
            threads = [
                threading.Thread(target=self.capture_video),
                threading.Thread(target=self.process_frames),
                threading.Thread(target=self.display_frames),
                threading.Thread(target=self.record_audio)
            ]
            
            for thread in threads:
                thread.daemon = True
                thread.start()
            
            # Wait for 'q' key press
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self.running = False
            cv2.destroyAllWindows()
            self.logger.info("Cleanup completed")

if __name__ == "__main__":
    try:
        system = BasicDetectionSystem()
        system.run()
    except Exception as e:
        logging.error(f"System error: {e}")