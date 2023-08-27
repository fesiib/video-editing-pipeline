import cv2
import json
import webvtt
import os
import numpy as np

from yt_dlp import YoutubeDL

from pathlib import Path

ROOT = Path('.')
DATABASE = 'videos'

video_library = {}

class VideoProcessor:
    def __init__(self, video_file) -> None:
        self.transcript = ""
        self.video_file = video_file
    
    def get_video_by_filename(filename):
        return DATABASE / filename
    
    def download_video(video_link):
        options = {
            'format': 'mp4[height<=480]',
            'outtmpl': os.path.join(DATABASE, '%(id)s.%(ext)s'),
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': {'en'},  # Download English subtitles
            'subtitlesformat': '/vtt/g',
            'skip_download': False,
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
            return 

    def process_video(video_link):
        print(f"Requested Link '{video_link}'")
        if (video_link not in video_library):
            video_library[video_link] = download_video(video_link)
        
        metadata = video_library[video_link]
        video_title = metadata.get('id')
        print(f"'{video_title}'")

        video_path = os.path.join(DATABASE, f'{video_title}.mp4')
        subtitles_path = os.path.join(DATABASE, f'{video_title}.en.vtt')
        
        transcript = []
        moments = []

        video_cap = cv2.VideoCapture(video_path)
        video_cap.release()
        
        subtitles = webvtt.read(subtitles_path)
        transcript = get_transcript(subtitles)
        return transcript, metadata

    def process_transcript(subtitles):
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

    def video_to_images(self, interval, output_folder):
        clip = mp.VideoFileClip(self.video_file)
        
        os.makedirs(output_folder, exist_ok=True)
    
        # total num frames + interval offset
        total_frames = int(clip.fps * clip.duration)
        frames_step = int(clip.fps * interval)
    
        vidcap = cv2.VideoCapture(self.video_file)
    
        for i in range(0, total_frames, frames_step):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, image = vidcap.read()
            if success:
                cv2.imwrite(os.path.join(output_folder, "frame_{}.jpg".format(i), image))
        return 0 

def main():
    video_file = "../data/linus_unboxing.mp4"
    video_processor = VideoProcessor(video_file)
    transcript, metadata = process_video(video_file)
    print(transcript)

if __name__=="__main__":
    main()