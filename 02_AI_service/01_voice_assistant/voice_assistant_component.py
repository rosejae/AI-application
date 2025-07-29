import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder


from dotenv import load_dotenv
load_dotenv()

# Init
client = OpenAI()

# Audio 레코드
audio_bytes = audio_recorder("talk", pause_threshold=1.0, auto_start=True)
if audio_bytes:
    with open("./tmp_audio.wav", "wb") as f:
        f.write(audio_bytes)

    with open("./tmp_audio.wav", "rb") as f: 
        # 음성 인식
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
        text = transcript.text
    st.write(f"transcription: {text}")

    # 음성 출력
    speech_file_path = "tmp_speak.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy", # alloy, echo, fable, onyx, nova, and shimmer
        input=text
    )
    response.stream_to_file(speech_file_path)
    st.audio(speech_file_path, format="audio/mpeg", loop=False, autoplay=True)