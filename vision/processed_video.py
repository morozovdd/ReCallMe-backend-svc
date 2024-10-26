import cv2
import threading
import queue
import time
import os
import subprocess
import datetime
from pathlib import Path

# Queues for frame processing
frame_queue = queue.Queue(maxsize=50)
processed_frame_queue = queue.Queue(maxsize=50)

running = True

def process_saved_video(video_file_path):
    global running
    cap = cv2.VideoCapture(video_file_path)  # Open the saved video file
    if not cap.isOpened():
        print(f"Error opening video file: {video_file_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties: {frame_width}x{frame_height} at {fps} FPS, {total_frames} total frames.")
    
    while running:
        ret, frame = cap.read()
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
        else:
            break  # End of video
    
    cap.release()

def frame_processing():
    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is not None:
                # Example processing: Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Here, you can add more processing logic (e.g., face detection)
                
                # For demonstration, put the processed frame into another queue
                processed_frame_queue.put(gray_frame)
            frame_queue.task_done()
        time.sleep(0.01)  # Prevent CPU overuse

def display_processed_frames():
    while running:
        if not processed_frame_queue.empty():
            processed_frame = processed_frame_queue.get()
            if processed_frame is not None:
                cv2.imshow("Processed Frame", processed_frame)  # Display processed frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    global running
                    running = False  # Exit on 'q' key press
            processed_frame_queue.task_done()
        time.sleep(0.01)  # Prevent CPU overuse

def main():
    video_file_path = "path_to_your_saved_video.mp4"  # Replace with your video file path
    
    # Start video capture thread
    capture_thread = threading.Thread(target=process_saved_video, args=(video_file_path,))
    capture_thread.start()
    
    # Start frame processing thread
    processing_thread = threading.Thread(target=frame_processing)
    processing_thread.start()
    
    # Start display thread
    display_thread = threading.Thread(target=display_processed_frames)
    display_thread.start()
    
    try:
        while running:
            time.sleep(0.1)  # Main thread idle
    except KeyboardInterrupt:
        global running
        running = False
    
    capture_thread.join()
    processing_thread.join()
    display_thread.join()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()