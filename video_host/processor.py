import cv2
import json
import webvtt
import os
import re

#import whisper
import difflib


from yt_dlp import YoutubeDL

from pathlib import Path

ROOT = Path('.')
DATABASE = ROOT / "static" / "database"

video_library = {}

def get_video_by_filename(filename):
    return DATABASE / filename

def download_video(video_link):
    options = {
        'format': 'best[height<=480][ext=mp4]',
		#'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=mp3]/best',
        'outtmpl': os.path.join(DATABASE, '%(id)s.%(ext)s'),
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': {'en'},  # Download English subtitles
        'subtitlesformat': '/vtt/g',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'keepvideo': True,
        'skip_download': False,


        # "paths": {
        #     "home": str(DATABASE)
        # },
        # "outtmpl": {
        #     "default": "%(id)s.%(ext)s"
        # },
        # "writesubtitles": True,
        # "writeautomaticsub": True,
        # "format": "mp4[height<=480]",
        # "subtitleslangs": {"en"},
        # "subtitlesformat": "/vtt/g",
		# "retries": 10,
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_link, download=False)
        metadata = ydl.sanitize_info(info)
        video_title = metadata.get('id')
        video_path = os.path.join(DATABASE, f'{video_title}.mp4')
        if not os.path.exists(video_path):
            ydl.download([video_link])
            print(f"Video '{video_title}' downloaded successfully.")
        else:
            print(f"Video '{video_title}' already exists in the directory.")
        return metadata

# def get_alternative_transcript(audio_path):
#     output_path = audio_path.replace(".mp3", ".alt.json")
#     if os.path.exists(output_path):
#         with open(output_path, 'r') as f:
#             return json.load(f)
#     model = whisper.load_model("small.en")
#     transcript = model.transcribe(audio_path)
#     with open(output_path, 'w') as f:
#         json.dump(transcript, f, indent=2)
#     return transcript

def extract_words(s):
    # Match words that are between >< or outside of any tags
    words = re.findall(r'>([^<]+)<|([^<>\s]+)', s)
    # Flatten the list of tuples and remove empty strings
    result = [word for tup in words for word in tup if word]
    return result
def format_transcript(transcript_file):
    result = []
    with open(transcript_file, 'r', encoding='utf-8') as f:
        transcript = f.read()
        # Split the transcript into line groups
        line_groups = transcript.strip().split("\n\n")
        # Filter out line groups that don't have a line containing '</c>'
        filtered_line_groups = [group for group in line_groups if any('</c>' in line for line in group.split("\n"))]
        filtered_transcript = ''
        for group in filtered_line_groups:
            first_line = group.split("\n")[0]
            last_line = group.split("\n")[2]
            pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})"
            match = re.search(pattern, first_line)
            if match:
                start_timecode = match.group(1)
                end_timecode = match.group(2)
                filtered_line = '<' + start_timecode + '>' + last_line + '<' + end_timecode + '>'
                cleaned_string = filtered_line.replace('<c>', '').replace('</c>', '')
                words = extract_words(cleaned_string)
                idx = 0
                while idx + 2 < len(words):
                    new_arr = words[idx:idx + 3]
                    timecoded_words = {"start": new_arr[0], "finish": new_arr[2], "text": new_arr[1]}
                    result.append(timecoded_words)
                    #print(timecoded_words)
                    idx += 2
            else:
                print("No timecodes found.")
    return result

def get_transcript(subtitles):
    transcript = []
    for caption in subtitles:
        lines = caption.text.strip("\n ").split("\n")
        if len(transcript) == 0:
            transcript.append({
                "start": caption.start,
                "finish": caption.end,
                "text": "\n".join(lines),
            })
            continue
        last_caption = transcript[len(transcript) - 1]

        new_text = ""
        for line in lines:
            if line.startswith(last_caption["text"], 0):
                new_line = line[len(last_caption["text"]):-1].strip()
                if len(new_line) > 0:
                    new_text += new_line + "\n"
            elif len(line) > 0:
                new_text += line + "\n"
        new_text = new_text.strip("\n ")
        if len(new_text) == 0:
            transcript[len(transcript) - 1]["finish"] = caption.end
        else:
            transcript.append({
                "start": caption.start,
                "finish": caption.end,
                "text": new_text,
            })
    return transcript

def get_moments(stream):
    while (True):
        res, frame = stream.read()
        if (res == False):
            break
    
    return [{
        "start": 1,
        "finish": 5,
        "transcriptStart": 0,
        "transcriptFinish": 1,
        "type": "text caption",
    }]

def git_difference(transcript, alternative_transcript):
    file_1 = ""
    file_2 = alternative_transcript["text"]
    for line in transcript:
        file_1 += line["text"] + " "
    print(file_1)
    print(file_2)
    diff = difflib.ndiff(file_1, file_2)
    print('\n'.join(diff))



def process_video(video_link):
    print(f"Requested Link '{video_link}'")
    if (video_link not in video_library):
        video_library[video_link] = download_video(video_link)
    
    metadata = video_library[video_link]
    video_title = metadata.get('id')
    print(f"'{video_title}'")

    video_path = os.path.join(DATABASE, f'{video_title}.mp4')
    subtitles_path = os.path.join(DATABASE, f'{video_title}.en.vtt')
    audio_path = os.path.join(DATABASE, f'{video_title}.mp3')
    
    transcript = []
    moments = []

    video_cap = cv2.VideoCapture(video_path)
    moments = get_moments(video_cap)
    video_cap.release()
    
    subtitles = webvtt.read(subtitles_path)
    
    transcript = get_transcript(subtitles)
    # alternative_transcript = get_alternative_transcript(audio_path)

    # git_difference(transcript, alternative_transcript)

    # transcript = format_transcript(subtitles_path)

    # transcript = get_transcript_each_word(subtitles)
    return transcript, moments, metadata