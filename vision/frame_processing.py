import cv2
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pathlib
import json

load_dotenv()

gemini_api = os.getenv("GEMINI")

genai.configure(api_key = gemini_api)

orb = cv2.ORB_create()
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def frame_processing(directory, output_directory):
    prev_descriptors = None
    os.makedirs(output_directory, exist_ok=True)
    for entry in os.scandir(directory):
        if entry.is_dir() and entry.name.endswith("video"):
            video_directory = os.path.join(directory, entry.name)
            print(f"Processing folder:{video_directory}")
            for filename in os.listdir(video_directory):
                print(filename)
                if filename.endswith(".avi") or filename.endswith(".mp4"):
                    video_path = os.path.join(video_directory, filename)
                    cap = cv2.VideoCapture(video_path)
                    frame_count = 0
                    video_name = os.path.splitext(filename)[0]
                    video_output_dir = os.path.join(output_directory, video_name)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        start_time = time.time()
                        faces = detect_bounding_box(frame)  
                        print(f"Number of faces:{len(faces)}")
                        if len(faces) > 0:
                            kp, des = extract_keypoints(frame=frame)
                            
                            if prev_descriptors is None:
                                save_frame(frame=frame, output_directory=video_output_dir, frame_count=frame_count)
                            else:
                                if not match_frames(des1=prev_descriptors, des2=des, min_match_count=200):
                                    save_frame(frame=frame, output_directory="/Users/dhruvmehrottra007/Desktop/Transformers", frame_count=frame_count)
                            prev_descriptors = des
                            frame_count += 1
                        end_time = time.time()    # End time after processing
                        print(f"Processing time: {end_time - start_time:.4f} seconds")
                    cap.release()  
                    print(f"Finished processing video: {filename}")


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 10)
    return faces

def extract_keypoints(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(frame, None)
    return kp, des

def match_frames(des1, des2, min_match_count):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    print(f"Number of matches: {len(matches)}")
    if len(matches) > min_match_count:
        return True
    return False

def save_frame(frame, output_directory, frame_count):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    cv2.imwrite(f"{output_directory}/keyframe_{frame_count}.jpg", frame)

def transcribe_audio(directory):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    transcript_directory = os.path.join(directory, "transcript")
    os.makedirs(transcript_directory, exist_ok=True)
    prompt = "Generate a transcript of the speech in the format of speaker, caption. Use speaker A, speaker B etc to identify speakers. When you transcribe please be professional in terms of the grammar and punctuation"
    for entry in os.scandir(directory):
        if entry.is_dir() and entry.name.endswith("audio"):
            audio_directory = os.path.join(directory, entry.name)
            for filename in os.listdir(audio_directory):
                if filename.endswith(".wav"):
                    audio_file = os.path.join(audio_directory, filename)
                    audio_base_name = os.path.splitext(filename)[0]
                    video_transcript_directory = os.path.join(transcript_directory, audio_base_name)
                    #print(video_transcript_directory)
                    os.makedirs(video_transcript_directory, exist_ok=True)
                    response = model.generate_content([prompt,{
                        "mime_type": "audio/wav",
                        "data": pathlib.Path(audio_file).read_bytes()
                    }])
                    transcript_filename = os.path.splitext(filename)[0] + ".json"
                    transcript_path = os.path.join(video_transcript_directory, transcript_filename)
                    print("Transcript_path:", transcript_path)
                    with open(transcript_path, "w") as json_file:
                        json.dump({"transcription": response.text}, json_file, indent=4)

                    print(f"Saved transcription to:{transcript_path}")
                    
video_directory = "content"
output_directory = "content/frames"
os.makedirs(video_directory, exist_ok=True)
#frame_processing(video_directory, output_directory)
transcribe_audio(directory=video_directory)