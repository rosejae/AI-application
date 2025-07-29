import streamlit as st
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder

from dotenv import load_dotenv
load_dotenv()

# Init
client = OpenAI()

if "messages" not in st.session_state:
    st.session_state.messages = []

# View
st.title("AI Voice Assistant")

con1 = st.container()
con2 = st.container()

user_input = ""

with con2:
    # 음성 인식
    audio_bytes = audio_recorder("talk", pause_threshold=3.0, auto_start= True)
    try:
        if audio_bytes:
            with open("./tmp_audio.wav", "wb") as f:
                f.write(audio_bytes)

            with open("./tmp_audio.wav", "rb") as f: 
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
                user_input = transcript.text
    except Exception as e:
        pass


with con1:
    # 메세지 히스토리
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # User의 현재 턴 메세지
        with st.chat_message("user"):
            st.markdown(user_input)

        # Assistant의 현재 턴 메세지
        with st.chat_message("assistant"):

            # Assistant의 현재 턴 응답 LLM으로 생성
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            assistant_text_response = st.write_stream(stream)

            # Assistant의 현재 턴 응답 텍스트를 Speech 로 만들기
            speech_file_path = "tmp_speak.mp3"
            assistant_speech_response = client.audio.speech.create(
                model="tts-1",
                voice="alloy", # alloy, echo, fable, onyx, nova, and shimmer
                input=assistant_text_response
            )
            assistant_speech_response.stream_to_file(speech_file_path)

            # 음성 플레이
            st.audio(speech_file_path, format="audio/mpeg", loop=False, autoplay=True)

        st.session_state.messages.append({"role": "assistant", "content": assistant_text_response})
