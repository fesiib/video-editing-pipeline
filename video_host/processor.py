import cv2
import json
import webvtt
import os
import re
import ast

import whisper
import difflib

from backend.helpers import Timecode

from yt_dlp import YoutubeDL

from pathlib import Path

ROOT = Path('.')
DATABASE = ROOT / "static" / "database"
METADATA = ROOT / "metadata"
SEGMENTATION_DATA = ROOT / "segmentation-data"

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

def yt_metadata(video_link):
    options = {
        'flat_playlist': True,
        'extract_flat': True,
        'quiet': False,  # Set to True if you want less console output
        'simulate': False,  # Set to True if you want to simulate the download
        'postprocessors': [{
            'key': 'ExecAfterDownload',
            'exec_cmd': 'pre_process:yt-dlp %(url)q --flat-playlist --print "Video - %(title)s - %(id)s"'
        }]
    }

    # Create the YouTubeDL object
    with YoutubeDL(options) as ydl:

        # Download the metadata for the main playlist
        result = ydl.extract_info(video_link, download=False)
        playlists = {}
        # Check if the extraction was successful
        if 'entries' in result:
            # Iterate over each playlist and download its metadata
            for entry in result['entries']:
                if 'url' in entry:
                    playlist_url = entry['url']
                    print(f"Processing Playlist: {playlist_url}")
                    
                    # Download the playlist metadata
                    metadata = ydl.download([playlist_url])
                    playlists[playlist_url] = metadata
        return playlists
    return {}

def get_alternative_transcript(audio_path):
    output_path = audio_path.replace(".mp3", ".alt.json")
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            return json.load(f)
    model = whisper.load_model("small.en")
    transcript = model.transcribe(audio_path)
    with open(output_path, 'w') as f:
        json.dump(transcript, f, indent=2)
    return transcript

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

def clip_metadata_file(new_title, old_title, suffix, clipStart, clipFinish):
    metadata_path = os.path.join(METADATA, f'{old_title}{suffix}.txt')
    new_metadata_path = os.path.join(METADATA, f'{new_title}{suffix}.txt')
    if not os.path.exists(metadata_path):
        return
    new_metadata = []
    with open(metadata_path, 'r') as f:
        raw_lines = f.readlines()
        for line in raw_lines:
            data_point = ast.literal_eval(line.rstrip())
            startTimecode = Timecode(data_point["start"])
            endTimecode = Timecode(data_point["end"])
            start = startTimecode.total_seconds()
            end = endTimecode.total_seconds()
            if start >= clipStart and end <= clipFinish:
                data_point["start"] = Timecode.convert_sec_to_timecode(start - clipStart)
                data_point["end"] = Timecode.convert_sec_to_timecode(end - clipStart)
                new_metadata.append(data_point)
        with open(new_metadata_path, 'w') as f:
            for data_point in new_metadata:
                f.write(json.dumps(data_point) + "\n")


def clip_segmentation_data(new_title, old_title, clipStart, clipFinish):
    segmentation_data_path = os.path.join(SEGMENTATION_DATA, f'{old_title}')
    new_segmentation_data_path = os.path.join(SEGMENTATION_DATA, f'{new_title}')
    if not os.path.exists(segmentation_data_path):
        return
    if not os.path.exists(new_segmentation_data_path):
        os.mkdir(new_segmentation_data_path)
    ### copy all the relevant folders and rename them
    for frame_sec in range(round(clipStart), round(clipFinish) + 1):
        folder_name = os.path.join(segmentation_data_path, "Frame.{}.0".format(frame_sec))
        new_folder_name = os.path.join(new_segmentation_data_path, "Frame.{}.0".format(frame_sec - round(clipStart)))
        os.system(f"cp -r {folder_name} {new_folder_name}")

def process_clipped_video(video_link, clipStart, clipFinish):
    print(f"Clipping '{video_link}'")

    clip_video_link = video_link + "_clipped"

    if (clip_video_link not in video_library):
        if (video_link not in video_library):
            video_library[video_link] = download_video(video_link)
    
        metadata = video_library[video_link]
        video_title = metadata.get('id')
        clip_video_title = video_title + "_clipped"
        clip_metadata = metadata.copy()
        clip_metadata["id"] = clip_video_title
        video_library[clip_video_link] = clip_metadata
        
        print(f"'{clip_video_title}'")

        video_path = os.path.join(DATABASE, f'{video_title}.mp4')
        subtitles_path = os.path.join(DATABASE, f'{video_title}.en.vtt')
        audio_path = os.path.join(DATABASE, f'{video_title}.mp3')

        clip_filename = video_path.replace(".mp4", f"_clipped.mp4")
        ### create clipped video in the same format, override if such file exists
        # os.system(f"ffmpeg -i {video_path} -ss {clipStart} -to {clipFinish}  -c copy -avoid_negative_ts make_zero -async 1 -y {clip_filename}")
        os.system(f'ffmpeg -i {video_path} -ss {clipStart} -to {clipFinish} -vf "setpts=PTS-STARTPTS" -af "asetpts=PTS-STARTPTS" -r 30 -c:v libx264 -c:a aac -strict experimental -y {clip_filename}')

        subtitles = webvtt.read(subtitles_path)
        clip_subtitles = webvtt.WebVTT()
        for caption in subtitles:
            if caption.start_in_seconds >= float(clipStart) and caption.end_in_seconds <= float(clipFinish):
                caption_start = Timecode.convert_sec_to_timecode(caption.start_in_seconds - float(clipStart)) + f".{caption.start.split('.')[1]}"
                caption_end = Timecode.convert_sec_to_timecode(caption.end_in_seconds - float(clipStart)) + f".{caption.end.split('.')[1]}"
                clip_caption = webvtt.Caption(
                    start=caption_start,
                    end=caption_end,
                    text=caption.text
                )
                clip_subtitles.captions.append(clip_caption)
        clip_subtitles.save(subtitles_path.replace(".en.vtt", "_clipped.en.vtt"))

        clip_audio_filename = audio_path.replace(".mp3", f"_clipped.mp3")
        os.system(f"ffmpeg -i {audio_path} -ss {clipStart} -to {clipFinish} -c copy -y {clip_audio_filename}")


        clip_metadata_file(clip_video_title, video_title, "_10", clipStart, clipFinish)
        clip_metadata_file(clip_video_title, video_title, "_5", clipStart, clipFinish)
        clip_metadata_file(clip_video_title, video_title, "_10_combined", clipStart, clipFinish)
        clip_metadata_file(clip_video_title, video_title, "_5_combined", clipStart, clipFinish)

        clip_segmentation_data(clip_video_title, video_title, clipStart, clipFinish)

    return process_video(clip_video_link)


def get_frame_by_timestamp(video_path, timestamp):
    print (video_path, timestamp)
    ## read mp4 video
    video_cap = cv2.VideoCapture(str(video_path))
    frame_rate = video_cap.get(cv2.CAP_PROP_FPS)
    frame = int(float(timestamp) * frame_rate)
    video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  
    res, frame = video_cap.read()
    if (res == False):
        return None
    ## convert to binary in jpeg format
    image_binary = cv2.imencode('.jpg', frame)[1].tobytes()
    return image_binary
