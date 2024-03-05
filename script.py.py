from pydub import AudioSegment
import math
import librosa
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from transformers import AutoModelForCTC, AutoProcessor
import subprocess
from pytube import YouTube
from googleapiclient.discovery import build

def sanitize_filename(name, replacement=" "):
    invalid_chars = '\\/:*?"<>|'
    for char in invalid_chars:
        name = name.replace(char, replacement)
    return name[:240]

def convert_mp4_to_wav(input_file, output_file):
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def download_audio_from_youtube(url, output_path='.'):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return audio_stream.download(output_path=output_path)


youtube = build('youtube', 'v3', developerKey='AIzaSyCcU1pgi8WKPFHqzflMTETTrVkoelmGfqE')


search_response = youtube.search().list(
    q='Gaeilge',
    part='id,snippet',
    maxResults=1,
    type='video',
).execute()


for search_result in search_response.get('items', []):
    video_id = search_result['id']['videoId']
    video_title = search_result['snippet']['title']
    channel_title = search_result['snippet']['channelTitle']
    video_url = f'https://www.youtube.com/watch?v={video_id}'

    sanitized_channel_title = sanitize_filename(channel_title)
    sanitized_video_title = sanitize_filename(video_title)

    output_path = os.path.join('./Result', sanitized_channel_title, sanitized_video_title)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    downloaded_file = download_audio_from_youtube(video_url, output_path)

    output_file_name = sanitized_video_title + '.wav'
    output_file_path = os.path.join(output_path, output_file_name)
    convert_mp4_to_wav(downloaded_file, output_file_path)

    print(f'Downloading audio for: {video_title} ({video_url})')


model_id = "kingabzpro/wav2vec2-large-xls-r-1b-Irish"
model = AutoModelForCTC.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

def process_segment(segment, segment_number, file_path_prefix):
    segment_path = f"{file_path_prefix}_segment_{segment_number}.wav"
    segment.export(segment_path, format="wav")
    audio, rate = librosa.load(segment_path, sr=16000)
    input_values = processor(audio, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values).logits
    transcription = processor.batch_decode(logits.numpy())[0]
    os.remove(segment_path)
    return segment_number, transcription

def segment_audio_and_process(file_path, segment_length_ms=30000, parallel_segments=2):
    audio = AudioSegment.from_file(file_path)
    duration_ms = len(audio)
    segments_count = math.ceil(duration_ms / segment_length_ms)
    file_path_prefix = os.path.splitext(file_path)[0]
    transcriptions = []

    with ThreadPoolExecutor(max_workers=parallel_segments) as executor:
        futures = {executor.submit(process_segment, audio[i * segment_length_ms: min((i + 1) * segment_length_ms, duration_ms)], i + 1, file_path_prefix): i for i in range(segments_count)}
        
        for future in as_completed(futures):
            segment_number, transcription = future.result()
            transcriptions.append((segment_number, transcription))

    
    transcriptions.sort(key=lambda x: x[0])
    with open(f"{file_path_prefix}_transcriptions.txt", "w") as f:
        for _, transcription in transcriptions:
            f.write(f"{transcription[0]}\n")

def process_directory(root_dir, segment_length_ms=30000, parallel_segments=2):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {file_path}")
                segment_audio_and_process(file_path, segment_length_ms=segment_length_ms, parallel_segments=parallel_segments)


process_directory("D:\Youtube\Result")