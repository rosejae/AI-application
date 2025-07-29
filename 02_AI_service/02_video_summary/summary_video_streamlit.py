import streamlit as st
import requests
import json
import time
import base64
import tempfile
import os

# FastAPI 서버 URL
FASTAPI_URL = "http://127.0.0.1:8000"  # FastAPI 서버의 URL을 적절히 변경

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("🎥 동영상 요약")
    uploaded_file = st.file_uploader("동영상 업로드", type=["mp4", "avi", "mov"])
    start_button = st.button("요약 시작")

if uploaded_file:
    st.video(uploaded_file)

if start_button and uploaded_file:
    with st.spinner("영상을 처리 중입니다... 잠시만 기다려주세요."):
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # FastAPI 서버로 파일 전송
            # files = {"file": open(tmp_file_path, "rb")}
            # response = requests.post(f"{FASTAPI_URL}/summarize-video/", files=files)
            
            with open(tmp_file_path, "rb") as f:
                files = {"file": f}
                response = requests.post(f"{FASTAPI_URL}/summarize-video/", files=files)
            
            if response.status_code == 200:
                data = response.json()
                
                with st.container(border=True):
                    st.subheader("요약")
                    st.markdown(data["text_summary"])
                    
                    audio_url = f"{FASTAPI_URL}{data['audio_summary']}"
                    st.audio(audio_url, format="audio/wav", autoplay=True)
                    
                    st.subheader("오디오 스크립트")
                    st.markdown(data["audio_transcript"])
            else:
                st.error("비디오 처리 중 오류가 발생했습니다.")
        finally:
            # 임시 파일 삭제
            os.remove(tmp_file_path)
