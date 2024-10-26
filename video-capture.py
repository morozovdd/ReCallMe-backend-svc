import cv2
import threading
import queue
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from datetime import datetime
import anthropic
import base64
from typing import List, Dict, Tuple
import textwrap

class TextWindowManager:
    def __init__(self, window_width: int = 800, window_height: int = 400):
        self.window_width = window_width
        self.window_height = window_height
        self.background_color = (30, 30, 30)  # Dark gray background
        self.text_color = (255, 255, 255)  # White text
        self.highlight_color = (70, 130, 180)  # Steel blue for highlights
        
        # Create window
        self.window = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        cv2.namedWindow('Scene Analysis', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Scene Analysis', window_width, window_height)
        
        # Text settings
        self.max_line_length = 60
        self.margin = 20
        self.line_height = 30
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2

    def create_text_box(self, text: str, start_y: int, is_highlight: bool = False) -> int:
        """Create a text box with proper wrapping and return the next y position"""
        # Wrap text
        wrapped_lines = textwrap.wrap(text, width=self.max_line_length)
        
        # Calculate text box height
        box_height = len(wrapped_lines) * self.line_height + self.margin
        
        # Draw background for highlighted text
        if is_highlight:
            cv2.rectangle(
                self.window,
                (self.margin - 10, start_y - 20),
                (self.window_width - self.margin + 10, start_y + box_height),
                self.highlight_color,
                -1
            )
        
        # Draw text lines
        current_y = start_y
        for line in wrapped_lines:
            cv2.putText(
                self.window,
                line,
                (self.margin, current_y),
                self.font,
                self.font_scale,
                self.text_color if not is_highlight else (255, 255, 255),
                self.thickness
            )
            current_y += self.line_height
        
        return current_y + self.margin

class DetectionSystemWithClaude:
    def __init__(self, claude_api_key: str):
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.setup_directories()
        self.setup_queues()
        self.setup_model(claude_api_key)
        
        # Initialize window manager
        self.text_window = TextWindowManager()
        
        # State management
        self.running = True
        self.frame_width = 416
        self.frame_height = 416
        self.fps = 30
    
    def list_available_cameras(self):
        """List all available camera indices."""
        index = 0
        arr = []
        while True:
            cap = cv2.VideoCapture(index)
            if not cap.read()[0]:
                break
            else:
                arr.append(index)
            cap.release()
            index += 1
        return arr
    
    def encode_image_base64(self, frame):
        """Convert CV2 frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def setup_directories(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"recordings_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        self.logger.info(f"Created output directory: {self.output_dir}")

    def setup_queues(self) -> None:
        self.frame_queue = queue.Queue(maxsize=5)
        self.display_queue = queue.Queue(maxsize=5)
        self.llm_queue = queue.Queue(maxsize=2)
        self.detected_objects_history = []
        self.last_llm_process_time = 0
        self.llm_cooldown = 3.0

    def setup_model(self, claude_api_key: str) -> None:
        self.model = YOLO('yolov8n.pt')
        self.device = 'cpu'
        self.claude = anthropic.Client(api_key=claude_api_key)
        self.latest_description = "Initializing scene analysis..."

    def generate_scene_description(self, frame: np.ndarray, detected_objects: List[Dict]) -> str:
        """Generate scene description using Claude with optimized prompting"""
        try:
            base64_image = self.encode_image_base64(frame)
            if not base64_image:
                return None
            
            # Get unique objects with counts
            recent_objects = [obj['name'] for obj in self.detected_objects_history[-10:]]
            object_counts = {obj: recent_objects.count(obj) for obj in set(recent_objects)}
            objects_summary = ", ".join(f"{obj} ({count})" for obj, count in object_counts.items())
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this scene for someone with dementia.
                            Current objects: {objects_summary}
                            
                            Provide a natural, conversational 2-3 sentence description:
                            1. Key objects and their relationship
                            2. Any safety concerns
                            3. Simple guidance if needed
                            
                            Keep it brief and clear."""
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]

            response = self.claude.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            self.logger.error(f"Claude processing error: {str(e)}")
            return "Unable to analyze scene at this moment."

    def capture_video(self):
        available_cameras = self.list_available_cameras()
        self.logger.info(f"Available cameras: {available_cameras}")
        
        if not available_cameras:
            self.logger.error("No cameras available")
            self.running = False
            return
        
        camera_index = available_cameras[0]
        cap = cv2.VideoCapture(camera_index)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
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
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    out.write(frame)
                    
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
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    if frame is None:
                        continue
                    
                    # Run YOLO inference
                    results = self.model(frame, device=self.device, conf=0.25)
                    
                    # Process detections
                    current_detections = []
                    annotated_frame = frame.copy()
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = self.model.names[cls]
                            
                            current_detections.append({
                                'name': name,
                                'confidence': conf,
                                'box': (x1, y1, x2, y2)
                            })
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{name} {conf:.2f}"
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )
                    
                    # Update detection history
                    self.detected_objects_history.extend(current_detections)
                    if len(self.detected_objects_history) > 50:
                        self.detected_objects_history = self.detected_objects_history[-50:]
                    
                    # Check if we should process with Claude
                    current_time = time.time()
                    if current_time - self.last_llm_process_time >= self.llm_cooldown:
                        if not self.llm_queue.full():
                            self.llm_queue.put((frame.copy(), current_detections))
                            self.last_llm_process_time = current_time
                    
                    if not self.display_queue.full():
                        self.display_queue.put(annotated_frame)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Frame processing error: {str(e)}")
                continue
            
            time.sleep(0.001)

    def claude_processing(self):
        """Optimized Claude processing thread"""
        while self.running:
            try:
                if not self.llm_queue.empty():
                    frame, detections = self.llm_queue.get(timeout=0.1)
                    description = self.generate_scene_description(frame, detections)
                    if description:
                        self.latest_description = description
                        self.text_window.update(description, self.detected_objects_history[-10:])
                        self.logger.debug(f"Updated description: {description[:50]}...")
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Claude thread error: {str(e)}")
            time.sleep(0.1)

    def run(self):
        try:
            threads = [
                threading.Thread(target=self.capture_video),
                threading.Thread(target=self.process_frames),
                threading.Thread(target=self.claude_processing)
            ]
            
            for thread in threads:
                thread.daemon = True
                thread.start()
            
            # Main display loop
            while self.running:
                try:
                    if not self.display_queue.empty():
                        frame = self.display_queue.get(timeout=0.1)
                        if frame is not None:
                            cv2.imshow('Assistant View', frame)
                            
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                        break
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Main loop error: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            self.running = False
            for thread in threads:
                thread.join(timeout=1.0)
            cv2.destroyAllWindows()
            self.logger.info("Cleanup completed")

if __name__ == "__main__":
    CLAUDE_API_KEY = "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"  # Replace with your API key
    try:
        system = DetectionSystemWithClaude(CLAUDE_API_KEY)
        system.run()
    except Exception as e:
        logging.error(f"System error: {e}")