import base64
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

gemini_api = os.getenv("GEMINI")

genai.configure(api_key = gemini_api)

def encode_image(image):
    with open(image, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def summarize(frames_directory, summary_directory):
    os.makedirs(summary_directory, exist_ok=True)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    for root,dirs,files in os.walk(frames_directory):
        for dir_name in dirs:
            video_frame_dir = os.path.join(root, dir_name)
            video_summary_dir = os.path.join(summary_directory, dir_name)
            os.makedirs(video_summary_dir, exist_ok=True)
            frame_summaries = []
            for frame_file in os.listdir(video_frame_dir):
                    frame_path = os.path.join(video_frame_dir, frame_file)
                    if frame_file.endswith(".jpg"):
                        encoded_image = encode_image(image=frame_path)
                        response = model.generate_content([
                        "Describe what's happening in this image. Be as detailed as possible.",
                            {
                                "mime_type": "image/jpg",  
                                "data": encoded_image
                            }
                            ])
                        frame_summaries.append({"frame": frame_file, "summary": response.text})
            summary_path = os.path.join(video_summary_dir, "summary.json")
            with open(summary_path, "w") as summary_file:
                 json.dump(frame_summaries, summary_file, indent=4)
            print(f"Summary for video {dir_name} saved to {summary_path}")

base_dir = os.getcwd()
frames_directory = os.path.join(base_dir, "content", "frames")
summary_directory = os.path.join(base_dir, "content", "summary")


