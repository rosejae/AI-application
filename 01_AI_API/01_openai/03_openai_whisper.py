from dotenv import load_dotenv

load_dotenv()

import os
import urllib
import ssl

from IPython.display import Audio
from pathlib import Path
from pydub import AudioSegment

from openai import OpenAI

client = OpenAI()

#
# preprocessing
#

earnings_call_remote_filepath = "https://cdn.openai.com/API/examples/data/EarningsCall.wav"
earnings_call_filepath = "data/EarningsCall.wav"

ssl._create_default_https_context = ssl._create_unverified_context
urllib.request.urlretrieve(earnings_call_remote_filepath, earnings_call_filepath)

def milliseconds_until_sound(sound, silence_threshold_in_decibels=-20.0, chunk_size=10):
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold_in_decibels and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def trim_start(filepath):
    path = Path(filepath)
    directory = path.parent
    filename = path.name
    audio = AudioSegment.from_file(filepath, format="wav")
    start_trim = milliseconds_until_sound(audio)
    trimmed = audio[start_trim:]
    new_filename = directory / f"trimmed_{filename}"
    trimmed.export(new_filename, format="wav")
    return trimmed, new_filename

def transcribe_audio(file, output_dir):
    audio_path = os.path.join(output_dir, file)
    with open(audio_path, 'rb') as audio_data:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_data,
            )
        return transcription.text

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)

#
# post-processing (after whisper)
#

# punctuation 후처리 LLM 
def punctuation_assistant(ascii_transcript):

    # system_prompt = """당신은 텍스트에 구두점을 추가하는 유용한 도우미입니다. 원래 단어를 보존하고 필요한 구두점만 삽입합니다.
    #                    예를 들어 마침표, 쉼표, 대문자, 달러 기호 또는 백분율 기호와 같은 기호 및 형식을 사용합니다.
    #                    제공된 컨텍스트만 사용합니다. 컨텍스트가 제공되지 않은 경우 ‘컨텍스트가 제공되지 않음’이라고 말합니다.\n"""
    
    system_prompt = """You are a helpful assistant that adds punctuation to text.
      Preserve the original words and only insert necessary punctuation such as periods,
     commas, capialization, symbols like dollar sings or percentage signs, and formatting.
     Use only the context provided. If there is no context provided say, 'No context provided'\n"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response

# 전체적으로 금융 용어에 대해 확인하고 고쳐줌 
def product_assistant(ascii_transcript):
    # system_prompt = """당신은 금융 상품을 전문으로 하는 지능형 도우미입니다.
    #                    당신의 임무는 실적 발표 회의록을 처리하여 모든 금융 상품 및 일반 금융 용어에 대한 참조가 올바른 형식으로 되어 있는지 확인하는 것입니다.
    #                    일반적으로 약어로 축약되는 금융 상품이나 일반 용어에 대해 전체 용어를 철자하고 약어를 괄호 안에 표시해야 합니다.
    #                    예를 들어, ‘401k’는 ‘401(k) 퇴직 저축 플랜’, ‘HSA’는 ‘건강 저축 계좌 (HSA)’, ‘ROA’는 ‘자산 수익률 (ROA)’, ‘VaR’는 ‘위험 가치 (VaR)’, ‘PB’는 ‘주가 대비 장부가치 (PB) 비율’로 변환되어야 합니다.
    #                    마찬가지로, 금융 상품을 나타내는 숫자 표현은 숫자 표현으로 변환하고, 뒤에 전체 이름을 괄호 안에 표시해야 합니다.
    #                    예를 들어, ‘five two nine’는 ‘529 (교육 저축 계획)’, ‘four zero one k’는 ’401(k) (퇴직 저축 플랜)’으로 변환됩니다.
    #                    그러나 일부 약어는 맥락에 따라 다른 의미를 가질 수 있다는 점에 유의해야 합니다 (예: ‘LTV’는 ‘대출 비율’ 또는 ‘고객 생애 가치’를 의미할 수 있습니다). 해당 용어가 무엇을 의미하는지 맥락에서 판단하고 적절한 변환을 적용해야 합니다.
    #                    특정 금융 상품을 나타내지 않는 숫자 표현 (예: ‘twenty three percent’)은 그대로 두어야 합니다. 당신의 역할은 텍스트에서 금융 상품 용어를 분석하고 조정하는 것입니다. 작업이 완료되면 조정된 회의록과 변경한 단어 목록을 작성하십시오."""
    
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
     financial products and common financial terms are in the correct format. For each
     financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
     transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)'
    , 'ROA' should be transformed to 'Return on Assets (ROA)', 'VaR' should be transformed to 'Value at Risk (VaR)'
, and 'PB' should be transformed to 'Price to Book (PB) ratio'. Similarly, transform spoken numbers representing 
financial products into their numeric representations, followed by the full name of the product in parentheses. 
For instance, 'five two nine' to '529 (Education Savings Plan)' and 'four zero one k' to '401(k) (Retirement Savings Plan)'.
 However, be aware that some acronyms can have different meanings based on the context (e.g., 'LTV' can stand for 
'Loan to Value' or 'Lifetime Value'). You will need to discern from the context which term is being referred to 
and apply the appropriate transformation. In cases where numerical figures or metrics are spelled out but do not 
represent specific financial products (like 'twenty three percent'), these should be left as is. Your role is to
 analyze and adjust financial product terminology in the text. Once you've done that, produce the adjusted 
 transcript and a list of the words you've changed"""
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response

#
# pre processing
#

# 비어 있는 부분을 없애줌 
trimmed_audio, trimmed_filename = trim_start(earnings_call_filepath)

trimmed_audio = AudioSegment.from_wav(trimmed_filename)  
one_minute = 1 * 60 * 1000  
start_time = 0 
i = 0  
output_dir_trimmed = "trimmed_earnings_directory"  

if not os.path.isdir(output_dir_trimmed):  
    os.makedirs(output_dir_trimmed)

# 오디오가 긴 경우를 대비해, 오디오를 1분 단위 파일로 나눠줌
while start_time < len(trimmed_audio):  
    segment = trimmed_audio[start_time:start_time + one_minute]  
    segment.export(os.path.join(output_dir_trimmed, f"trimmed_{i:02d}.wav"), format="wav")  
    start_time += one_minute  
    i += 1  

# 정렬 
audio_files = sorted(
    (f for f in os.listdir(output_dir_trimmed) if f.endswith(".wav")),
    key=lambda f: int(''.join(filter(str.isdigit, f)))
)

transcriptions = [transcribe_audio(file, output_dir_trimmed) for file in audio_files]
full_transcript = ' '.join(transcriptions)

#
# post processing
#

ascii_transcript = remove_non_ascii(full_transcript)
response = punctuation_assistant(ascii_transcript)

punctuated_transcript = response.choices[0].message.content
response = product_assistant(punctuated_transcript)

final_transcript = response.choices[0].message.content

for line in final_transcript.split('. '):
    print(line)