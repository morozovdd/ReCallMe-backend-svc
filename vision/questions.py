import json
import os
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import time
from functools import wraps

import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
    retry_if_exception,
)
from ratelimit import limits, sleep_and_retry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI")
if not GEMINI_API_KEY:
    logger.error("GEMINI API key not found in environment variables.")
    raise EnvironmentError("GEMINI API key is required.")

# Configure Generative AI model
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name=MODEL_NAME)

# Define directories using pathlib
BASE_DIR = Path.cwd()
CONTENT_DIR = BASE_DIR / "content"
FRAMES_DIR = CONTENT_DIR / "frames"
TRANSCRIPT_DIR = CONTENT_DIR / "transcript"
SUMMARY_DIR = CONTENT_DIR / "summary"
VIDEOS_DIR = CONTENT_DIR / "video"
COMBINED_SUMMARY_DIR = CONTENT_DIR / "combined_summary"
QUESTIONS_ANSWERS_FILE = CONTENT_DIR / "questions_and_answers.json"

# Ensure output directories exist
COMBINED_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Define rate limiting parameters
MAX_CALLS = 5  # Adjust based on your quota
PERIOD = 60    # seconds

def load_json(file_path: Path) -> dict:
    """Load JSON data from a file."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded data from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in file {file_path}: {e}")
        return {}

def save_json(data: dict, file_path: Path) -> None:
    """Save JSON data to a file."""
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Saved data to {file_path}")
    except IOError as e:
        logger.error(f"IO error when writing to {file_path}: {e}")

def load_summaries_and_transcriptions(video_name: str) -> Tuple[List[Dict], str]:
    """Load frame summaries and transcription text for a given video."""
    summary_path = SUMMARY_DIR / video_name / "summary.json"
    transcription_path = TRANSCRIPT_DIR / video_name / f"{video_name}.json"

    frame_summaries = load_json(summary_path)
    transcription_data = load_json(transcription_path)
    transcription_text = transcription_data.get("transcription", "")

    return frame_summaries, transcription_text

def is_rate_limit_error(exception):
    """Check if the exception is a rate limit error (HTTP 429)."""
    if isinstance(exception, Exception):
        # Adjust based on how the Gemini API client represents errors
        if '429' in str(exception):
            return True
    return False

# Decorator to handle rate limiting and retries
@sleep_and_retry
@limits(calls=MAX_CALLS, period=PERIOD)
@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def consolidate_frame_summaries(frame_summaries: List[Dict]) -> str:
    """Combine frame summaries into a cohesive summary using the Generative AI model."""
    frame_texts = [frame.get("summary", "") for frame in frame_summaries]
    combined_text = " ".join(frame_texts)
    prompt = f"Combine the descriptions into a summary of the video: {combined_text}."

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=500,
            temperature=0.3,
        )
    )
    cohesive_summary = response.text.strip()
    logger.debug("Generated cohesive summary.")
    return cohesive_summary

@sleep_and_retry
@limits(calls=MAX_CALLS, period=PERIOD)
@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def combine_summaries(summary: str, transcription: str, video_name: str) -> None:
    """Combine the cohesive summary and transcription, then save to a JSON file."""
    prompt = (
        f"Can you combine the summary and the transcription to describe what is happening in the scene. "
        f"Video description: {summary}. Transcription: {transcription}"
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=100,
            temperature=0.7,
        )
    )
    cohesive_combined_summary = response.text.strip()
    logger.debug("Generated combined summary.")

    summary_filename = "combined_summary.json"
    summary_file_path = COMBINED_SUMMARY_DIR / summary_filename

    # Load existing data or initialize
    if summary_file_path.exists():
        combined_data = load_json(summary_file_path)
        if not isinstance(combined_data, list):
            logger.warning(f"Expected list in {summary_file_path}, initializing new list.")
            combined_data = []
    else:
        combined_data = []

    # Append new summary
    combined_data.append({
        "video_name": video_name,
        "combined_summary": cohesive_combined_summary
    })

    save_json(combined_data, summary_file_path)
    logger.info(f"Saved combined summary for {video_name} to: {summary_file_path}")

@sleep_and_retry
@limits(calls=MAX_CALLS, period=PERIOD)
@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def generate_questions(summary: str) -> str:
    """Generate a question based on the summary."""
    prompt = (
        f"Based on the following summary, generate an answerable question that's supposed to jog the recorder's memory. "
        f"Summary: {summary}"
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=20,
            temperature=0.7,
        )
    )
    question = response.text.strip()
    logger.debug("Generated question.")
    return question

@sleep_and_retry
@limits(calls=MAX_CALLS, period=PERIOD)
@retry(
    retry=retry_if_exception(is_rate_limit_error),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    reraise=True
)
def generate_answer(question: str, summary: str) -> str:
    """Generate an answer based on the question and summary."""
    prompt = (
        f"Given the summary of a video and a question, predict a answer from the query. Do not go over 15 words, make it concise. "
        f"Summary: {summary} Question: {question}"
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            stop_sequences=["x"],
            max_output_tokens=20,
            temperature=0.7,
        )
    )
    answer = response.text.strip()
    logger.debug("Generated answer.")
    return answer

def generate_questions_json() -> None:
    """Generate questions and answers for all combined summaries and save to a JSON file."""
    combined_summary_path = COMBINED_SUMMARY_DIR / "combined_summary.json"

    combined_data = load_json(combined_summary_path)
    if not isinstance(combined_data, list):
        logger.error(f"Invalid format in {combined_summary_path}. Expected a list.")
        return

    questions_and_answers = []

    for entry in combined_data:
        video_name = entry.get("video_name", "unknown_video")
        combined_summary = entry.get("combined_summary", "")
        if not combined_summary:
            logger.warning(f"No combined summary for {video_name}, skipping.")
            continue

        try:
            question = generate_questions(combined_summary)
            answer = generate_answer(question, combined_summary)
            questions_and_answers.append({
                "question": question,
                "video": f"video_{video_name}.mp4",
                "answer": answer
            })
        except Exception as e:
            logger.warning(f"Failed to generate Q&A for {video_name}: {e}")
            continue

    save_json(questions_and_answers, QUESTIONS_ANSWERS_FILE)
    logger.info(f"Saved questions and answers to: {QUESTIONS_ANSWERS_FILE}")

def process_video(video_file: Path) -> None:
    """Process a single video file."""
    if not video_file.is_file() or video_file.suffix.lower() != ".mp4":
        logger.debug(f"Skipping non-mp4 file: {video_file}")
        return

    video_name = video_file.stem
    logger.info(f"Processing video: {video_name}")

    summaries, transcription = load_summaries_and_transcriptions(video_name)
    if not summaries:
        logger.warning(f"No summaries found for {video_name}, skipping.")
        return
    if not transcription:
        logger.warning(f"No transcription found for {video_name}, skipping.")
        return

    try:
        cohesive_summary = consolidate_frame_summaries(summaries)
        if not cohesive_summary:
            logger.warning(f"Failed to create cohesive summary for {video_name}, skipping.")
            return

        combine_summaries(cohesive_summary, transcription, video_name)
    except Exception as e:
        logger.error(f"Error processing {video_name}: {e}")

def process_all_videos() -> None:
    """Process all video files in the videos directory and generate questions and answers."""
    video_files = list(VIDEOS_DIR.glob("*.mp4"))
    if not video_files:
        logger.warning(f"No video files found in {VIDEOS_DIR}")
        return

    for video_file in video_files:
        process_video(video_file)

    try:
        generate_questions_json()
    except Exception as e:
        logger.error(f"Error generating questions and answers: {e}")

if __name__ == "__main__":
    process_all_videos()