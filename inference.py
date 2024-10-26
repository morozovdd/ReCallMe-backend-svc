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

class CloudDetectionSystem:
    """
    Cloud-ready object detection and scene description system.
    Processes video frames, detects objects, and generates scene descriptions.
    Designed for headless operation in cloud environment.
    """
    
    def __init__(self, claude_api_key: str, input_source: str = None):
        """
        Initialize the detection system.
        
        Args:
            claude_api_key (str): API key for Claude
            input_source (str): Path to video file or camera index
        """
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Core components initialization
        self.setup_model(claude_api_key)
        self.setup_queues()
        self.input_source = input_source
        
        # Processing settings
        self.frame_width = 416
        self.frame_height = 416
        self.fps = 30
        self.running = True
        
        # Cloud-specific settings
        self.output_buffer = queue.Queue(maxsize=100)  # Buffer for processed frames
        self.scene_descriptions = queue.Queue(maxsize=100)  # Buffer for scene descriptions

    def setup_model(self, claude_api_key: str) -> None:
        """Initialize YOLO and Claude models"""
        self.model = YOLO('yolov8n.pt')
        self.device = 'cpu'  # Can be modified for cloud GPU
        self.claude = anthropic.Client(api_key=claude_api_key)

    def setup_queues(self) -> None:
        """Initialize processing queues"""
        self.frame_queue = queue.Queue(maxsize=30)  # Raw frame buffer
        self.detection_queue = queue.Queue(maxsize=30)  # Detection results buffer
        self.detected_objects_history = []
        self.last_llm_process_time = 0
        self.llm_cooldown = 5.0  # Seconds between Claude API calls

    def encode_frame_for_claude(self, frame: np.ndarray) -> str:
        """Convert frame to base64 for Claude API"""
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8')

    def generate_scene_description(self, frame: np.ndarray, detected_objects: List[Dict]) -> str:
        """Generate scene description using Claude"""
        try:
            base64_image = self.encode_frame_for_claude(frame)
            if not base64_image:
                return None
            
            # Prepare object summary
            recent_objects = [obj['name'] for obj in self.detected_objects_history[-10:]]
            object_counts = {obj: recent_objects.count(obj) for obj in set(recent_objects)}
            objects_summary = ", ".join(f"{obj} ({count})" for obj, count in object_counts.items())
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this scene concisely in 1-2 sentences.
                            Currently visible: {objects_summary}
                            Focus on: location of objects, any safety concerns, simple guidance."""
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
            return None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process single frame with YOLO detection"""
        try:
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
                    
                    # Draw bounding boxes
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated_frame,
                        f"{name} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            
            return annotated_frame, current_detections
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {str(e)}")
            return frame, []

    def frame_processing_worker(self) -> None:
        """Worker thread for processing frames"""
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    if frame is None:
                        continue
                    
                    # Process frame
                    annotated_frame, detections = self.process_frame(frame)
                    
                    # Update detection history
                    self.detected_objects_history.extend(detections)
                    if len(self.detected_objects_history) > 50:
                        self.detected_objects_history = self.detected_objects_history[-50:]
                    
                    # Add to output buffer
                    if not self.output_buffer.full():
                        self.output_buffer.put({
                            'frame': annotated_frame,
                            'detections': detections,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Check if we should process with Claude
                    current_time = time.time()
                    if current_time - self.last_llm_process_time >= self.llm_cooldown:
                        description = self.generate_scene_description(frame, detections)
                        if description:
                            self.scene_descriptions.put({
                                'description': description,
                                'timestamp': datetime.now().isoformat()
                            })
                            self.last_llm_process_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing worker error: {str(e)}")
            
            time.sleep(0.001)

    def get_latest_results(self) -> Dict:
        """
        Get latest processing results for API endpoint
        Returns dict with latest frame, detections, and description
        """
        try:
            latest_frame_data = self.output_buffer.get_nowait()
            latest_description = self.scene_descriptions.get_nowait()
            
            return {
                'frame': latest_frame_data['frame'],
                'detections': latest_frame_data['detections'],
                'description': latest_description['description'],
                'timestamp': latest_frame_data['timestamp']
            }
        except queue.Empty:
            return None

    def start(self) -> None:
        """Start the processing system"""
        self.processing_thread = threading.Thread(target=self.frame_processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Processing system started")

    def stop(self) -> None:
        """Stop the processing system"""
        self.running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)
        self.logger.info("Processing system stopped")

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)