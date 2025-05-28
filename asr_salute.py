import soundfile as sf
import numpy as np
import librosa

import requests as rq
import uuid
import base64
import pyaudio
import os
import webrtcvad
from flask import Flask, Blueprint, request, render_template
from flask_cors import CORS

# токен доступа для распознавания речи
AUTH_SALUTE = os.getenv("AUTH_SALUTE") or "NjBlMmIzNGQtYTc2OC00ZGY5LTg5NTQtMmFkNjRlZTdlNTIzOjRjODU3YmZiLWZkMDEtNDczZS04MWZiLTQ3ZGU2NTQyMWY1NA=="
# NEW TOKEN:
# AUTH_SALUTE = os.getenv("AUTH_SALUTE") or "N2RmYzQwZGMtYjQ0ZS00YzU2LWFhY2EtNWFmYjdhOGEzZDU1OjY5NzMzODIxLWQyMzMtNGQxMS05MzYwLWY0YmM4NGU5YWUxMg=="
CLIENT_ID = "7dfc40dc-b44e-4c56-aaca-5afb7a8a3d55"
class SpeechToText:
    def __init__(self, auth_token=AUTH_SALUTE):
        self.auth_token = auth_token
        self.salute_token = None

    def get_token(self, scope='SALUTE_SPEECH_PERS'):

        rq_uid = str(uuid.uuid4())
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'RqUID': rq_uid,
            'Authorization': f'Basic {self.auth_token}'
        }
        payload = {'scope': scope}
        try:
            response = rq.post(url, headers=headers, data=payload, verify=False)
            response.raise_for_status()
            return response
        except rq.RequestException as e:
            print(f"Ошибка получения токена: {str(e)}")
            return None

    def speech_to_text(self, voice) -> str:

        if self.salute_token is None:
            response = self.get_token()
            if response is not None:
                self.salute_token = response.json()['access_token']
            else:
                return "Не удалось получить токен"

        url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
        headers = {
            "Authorization": f"Bearer {self.salute_token}",
            "Content-Type": "audio/x-pcm;bit=16;rate=16000"
        }
        # The API expects raw PCM audio. Our function received a Base64 string so we decode it.
        audio_data = base64.b64decode(voice)
        try:
            response = rq.post(url, headers=headers, data=audio_data, verify=False)
            if response.status_code == 200:
                result = response.json()
                print("Полный ответ API:", result)
                return ''.join(result.get("result", "Результат не найден"))
            else:
                print("Ошибка распознавания:", response.status_code, response.text)
                return "Ошибка распознавания"
        except rq.RequestException as e:
            print("Ошибка при запросе:", str(e))
            return "Ошибка при запросаб см. лог-файл"


asr_client = SpeechToText(auth_token=AUTH_SALUTE)
wait_question = True

audio_chunks = []

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
SR = 16000
SAMPLE_WIDTH = 2
CHUNK = 960  # 60 мс аудио (16000 * 0.06 = 960)
RECORD_SECONDS = 2.4
WAVE_OUTPUT_FILENAME = "output.wav"

vad = webrtcvad.Vad(3)  # Агрессивность 0-3, где 3 – самое строгое определение речи
silence_in_seconds = 2.4
started = False
timeout_chunks = int(RATE / CHUNK * 2 * silence_in_seconds)  # по умолчанию 2.4 секунды без речи для завершения
silence_chunks = 0
# Set the frame duration for VAD analysis
frame_duration = 20  # in milliseconds
# Convert the frame duration to the number of samples
frame_size = int(SR * (frame_duration / 1000.0))

def recognize_speach(speach) -> str:
    question_path = './question.wav'
    with open(question_path, 'wb') as f:
        f.write(speach)
    audio, sr = librosa.load(question_path, sr=SR, dtype=np.int16)
    # speach_fragments = extract_speech_segments(question_path, False)
    # text_fragments = []
    # for fragment in speach_fragments:
    #     audio_b64 = base64.b64encode(fragment).decode("utf-8")
    #     text_fragments.append(asr_client.speech_to_text(audio_b64))
    # text = ''.join(text_fragments)

    audio_segment = b''.join(np.int16(audio * 32768))
    audio_b64 = base64.b64encode(audio_segment).decode("utf-8")
    text = asr_client.speech_to_text(audio_b64)
    return text


def extract_speech_segments(audio, save_segments:bool = False, output_path: str = 'question'):
    # audio, sr = librosa.load(input_path, sr = SR, dtype=np.int16)
    # audio, sr = librosa.load(input_path, sr = SR, mono=True, dtype=np.int16)
    segments = []
    current_segment_start = 0
    current_segment_end = 0
    for i in range(0, len(audio), frame_size):
        frame = audio[i:i + frame_size]
        frame = np.int16(frame * 32768)
        if vad.is_speech(frame.tobytes(), sample_rate=SR):
            if current_segment_start == 0:
                current_segment_start = i
            current_segment_end = i + frame_size
        else:
            if current_segment_start != 0:
                segments.append((current_segment_start, current_segment_end))
                current_segment_start = 0
                current_segment_end = 0

    speach_segments = []
    for idx, (start, end) in enumerate(segments):
        audio_segment = np.int16(audio[start:end] * 32768)
        speach_segments.append(b''.join(audio_segment))
        if save_segments:
            segment_output_path = f"{output_path}.segment{idx}.wav"
            sf.write(segment_output_path, audio_segment, SR)

    return speach_segments

def pcm2wav(sample_rate, pcm_voice):
    # if pcm_voice.startswith("RIFF".encode()):
    #     return pcm_voice
    # else:
    import struct
    sampleNum = len(pcm_voice)
    rHeaderInfo = "RIFF".encode()
    rHeaderInfo += struct.pack('i', sampleNum + 44)
    rHeaderInfo += 'WAVEfmt '.encode()
    rHeaderInfo += struct.pack('i', 16)
    rHeaderInfo += struct.pack('h', 1)
    rHeaderInfo += struct.pack('h', 1)
    rHeaderInfo += struct.pack('i', sample_rate)
    rHeaderInfo += struct.pack('i', sample_rate * int(32 / 8))
    rHeaderInfo += struct.pack("h", int(32 / 8))
    rHeaderInfo += struct.pack("h", 32)
    rHeaderInfo += "data".encode()
    rHeaderInfo += struct.pack('i', sampleNum)
    rHeaderInfo += pcm_voice
    return rHeaderInfo
