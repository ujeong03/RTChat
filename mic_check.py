# import pyaudio
# import numpy as np

# # 설정
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000  # 샘플링 레이트
# CHUNK = 1024  # 한 번에 처리할 오디오 샘플의 크기
# RECORD_SECONDS = 5  # 녹음 시간
# OUTPUT_FILENAME = "output.wav"  # 저장할 파일명

# # pyaudio 초기화
# p = pyaudio.PyAudio()

# # 마이크에서 입력받기
# stream = p.open(format=FORMAT,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 frames_per_buffer=CHUNK)

# print("Recording...")

# frames = []

# for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#     data = stream.read(CHUNK)
#     frames.append(data)

# # 스트림 종료
# print("Finished recording.")
# stream.stop_stream()
# stream.close()
# p.terminate()

# # 음성 데이터를 numpy 배열로 변환
# audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)

# # 샘플 레이트는 RATE로 설정
# sample_rate = RATE

# # audio와 sample_rate 출력
# print(f"Audio data: {audio_data[:10]}...")  # 앞 부분 10개만 출력
# print(f"Sample rate: {sample_rate}")

# # 저장할 파일로 저장
# import soundfile as sf
# sf.write(OUTPUT_FILENAME, audio_data, sample_rate)

import ctypes
ctypes.CDLL("c:/Users/yhim0/anaconda3/envs/rt_diary/Lib/site-packages/torio/lib/libtorio_ffmpeg6.pyd")

