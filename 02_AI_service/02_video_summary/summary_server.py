import os
import shutil
import tempfile
import base64
import uuid
import vertexai

from vertexai.generative_models import GenerativeModel, Part
from moviepy import VideoFileClip
from openai import OpenAI
from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
client = OpenAI()

TEMP_DIR = "./tmp_video_segments"
OUTPUT_DIR = "./outputs"

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

ensure_dir(TEMP_DIR)
ensure_dir(OUTPUT_DIR)

# Segmented 된 비디오 하나가 들어오면 요약을 해줌
def summarize_segment(video_segment, previous_summary="", segment_info=""):
    model = GenerativeModel("gemini-2.5-pro")
    context = f"이전 구간 요약: {previous_summary}\n\n현재 구간 정보: {segment_info}\n\n"
    prompt = context + "이 영상 구간을 앞선 맥락을 고려하여 bullet point 사용해서 2-3줄로 요약해줘. 중복되는 내용은 제외하고, 새로운 정보나 변화에 초점을 맞춰주세요."
    
    responses = model.generate_content(
        [video_segment, prompt],
        generation_config={
            "max_output_tokens": 2048,
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32
        },
    )
    return responses.text

# 전체 비디오를 일정한 간격으로 segment 시켜줌
def process_video(video_path, segment_duration=60, overlap_duration=10):
    clip = VideoFileClip(video_path)
    total_duration = clip.duration
    segments = []
    temp_files = []

    for i, start_time in enumerate(range(0, int(total_duration), segment_duration - overlap_duration)):
        end_time = min(start_time + segment_duration, total_duration)
        segment = clip.subclipped(start_time, end_time)
        
        # 동시에 요청을 받을 수도 있는데, 중간에 생성되는 임시 파일의 이름이 중복되지 않게 하려면
        # 임시 파일이 유니크한 이름을 가지고 있어야함 -> uuid.uuid4()
        temp_file_path = os.path.join(TEMP_DIR, f"segment_{i}_{uuid.uuid4()}.mp4")
        segment.write_videofile(temp_file_path, codec="libx264")
        temp_files.append(temp_file_path)
        
        with open(temp_file_path, "rb") as f:
            video_encoded = base64.b64encode(f.read()).decode('utf-8')
        video_part = Part.from_data(data=video_encoded, mime_type="video/mp4")
        segments.append((start_time, end_time, video_part))
        
        if end_time == total_duration:
            break

    clip.close()
    return segments, temp_files

# 각자 요약된 비디오를 참고해서 5개의 bullet point로 바꿔줌 
def summarize_video(video_path, segment_duration=60, overlap_duration=10):
    segments, temp_files = process_video(video_path, segment_duration, overlap_duration)
    summaries = []
    previous_summary = ""

    try:
        for i, (start_time, end_time, segment) in enumerate(segments):
            segment_info = f"구간 {i+1} ({start_time:.1f}s - {end_time:.1f}s)"
            summary = summarize_segment(segment, previous_summary, segment_info)
            summaries.append(f"{segment_info}:\n{summary}\n")
            previous_summary = summary

        full_summary = "\n".join(summaries)
        model = GenerativeModel("gemini-2.5-pro")
        final_summary = model.generate_content(
            [f"다음은 비디오의 각 구간별 요약입니다. 이를 바탕으로 전체 비디오의 내용을 5개의 bullet point로 요약해주세요. 중복을 피하고 주요 내용과 변화에 초점을 맞춰주세요:\n\n{full_summary}"],
            generation_config={
                "max_output_tokens": 2048,
                "temperature": 0.4,
                "top_p": 1,
                "top_k": 32
            },
        ).text

        return final_summary, summaries

    finally:
        pass
        # 임시 파일을 지우고 싶다면 주석 해제
        # for temp_file in temp_files:
        #     try:
        #         os.remove(temp_file)
        #     except Exception as e:
        #         print(f"임시 파일 삭제 중 오류 발생: {e}")

# 문서로 작성된 요약본을 자연스럽게 듣기 좋은 형식으로 바꿔줌
def create_audio_summary(summary):
    prompt = f"{summary}\n위 내용을 전달하기 위해 사용자가 듣기 편한 짧막한 오디오 대본을 만들어줘. 너의 응답을 그대로 TTS에 넣을 것이니 자연스러운 발화만 이야기해줘. 뉴스를 전하는 톤으로 존댓말을 사용해서 만들어줘."
    
    model = GenerativeModel("gemini-2.5-pro")
    natural_summary = model.generate_content(prompt).text

    unique_filename = f"audio_summary_{uuid.uuid4()}.wav"
    speech_file_path = os.path.join(OUTPUT_DIR, unique_filename)
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=natural_summary
    )
    response.stream_to_file(speech_file_path)
    
    return speech_file_path, natural_summary

#
# Fast API 구현
#

class SummaryResponse(BaseModel):
    text_summary: str
    audio_summary: str
    audio_transcript: str

@app.post("/summarize-video/", response_model=SummaryResponse)
async def summarize_video_endpoint(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        shutil.copyfileobj(file.file, temp_video)
        temp_video_path = temp_video.name

    try:
        final_summary, _ = summarize_video(temp_video_path)
        audio_file_path, audio_transcript = create_audio_summary(final_summary)

        return SummaryResponse(
            text_summary=final_summary,
            audio_summary=f"/audio-summary/{os.path.basename(audio_file_path)}",
            audio_transcript=audio_transcript
        )
    finally:
        os.remove(temp_video_path)

@app.get("/audio-summary/{filename}")
async def get_audio_summary(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(file_path, media_type="audio/wav", filename=filename)