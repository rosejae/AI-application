from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

client = OpenAI()

tts = client.audio.speech.create(
    model="tts-1",
    voice="nova", # alloy, echo, fable, , nova, and shimmer
    input="안녕하세요 강사 리암입니다."
)

speech_file_path = r"outputs\sample_speech.mp3"
tts.stream_to_file(speech_file_path)

from IPython.display import Audio

# Audio 객체를 생성하고 출력합니다.
# audio = Audio(speech_file_path, autoplay=False)
# audio


