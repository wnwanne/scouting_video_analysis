<<<<<<< HEAD
# # Import libraries
# import streamlit as st
# import cv2
# import os
# import time
# import json
# from dotenv import load_dotenv
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
# from moviepy.editor import VideoFileClip
# from openai import AzureOpenAI
# import base64
# import yt_dlp
# from yt_dlp.utils import download_range_func


# # Streamlit User Interface
# st.set_page_config(
#     page_title="Video Analysis with GPT-4.1",
#     layout="centered",
#     initial_sidebar_state="auto",
# )
# st.image("microsoft.png", width=100)
# st.title('Video Analysis with GPT-4.1')

# # Default configuration
# DEFAULT_SHOT_INTERVAL = 30  # In seconds
# DEFAULT_FRAMES_PER_SECOND = 1
# system_prompt = """"You are an expert NFL scouting assistant. You will be shown a series of video frames (and transcriptions if available)
#     from a football game or practice. Analyze and describe the player's physical attributes, athleticism, technique, positional skills,
#     decision-making, and any notable plays. Highlight strengths, weaknesses, and potential fit for specific NFL roles.
#     Be objective, concise, and use scouting terminology. If possible, compare the player's traits to current or former NFL athletes."""
# USER_PROMPT = "These are the frames from the video. Focus your analysis on player evaluation and scouting insights."
# DEFAULT_TEMPERATURE = 0.5
# RESIZE_OF_FRAMES = 4  # Changed default resize ratio to 4

# # Load configuration
# load_dotenv(override=True)

# # Configuration of OpenAI GPT-4o
# # aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
# # aoai_apikey = os.environ["AZURE_OPENAI_API_KEY"]
# # aoai_apiversion = os.environ["AZURE_OPENAI_API_VERSION"]
# # aoai_model_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
# # system_prompt = os.environ.get("SYSTEM_PROMPT", "You are an expert on Video Analysis. You will be shown a series of images from a video. Describe what is happening in the video, including the objects, actions, and any other relevant details. Be as specific and detailed as possible.")
# # print(f'aoai_endpoint: {aoai_endpoint}, aoai_model_name: {aoai_model_name}')
# # Create AOAI client for answer generation
# with st.sidebar:
#     st.header("Azure OpenAI Configuration")
#     aoai_endpoint = st.text_input("Azure OpenAI Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
#     aoai_apikey = st.text_input("Azure OpenAI API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
#     aoai_apiversion = st.text_input("Azure OpenAI API Version", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"))
#     aoai_model_name = st.text_input("AOAI Model Name/Deployment", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"))
#     st.markdown("---")
#     st.header("Whisper Speech-to-Text Configuration")
#     whisper_endpoint = st.text_input("Whisper Endpoint", value=os.getenv("WHISPER_ENDPOINT", ""))
#     whisper_apikey = st.text_input("Whisper API Key", value=os.getenv("WHISPER_API_KEY", ""), type="password")
#     whisper_apiversion = st.text_input("Whisper API Version", value=os.getenv("WHISPER_API_VERSION", "2024-05-01-preview"))
#     whisper_model_name = st.text_input("Whisper Deployment Name", value=os.getenv("WHISPER_DEPLOYMENT_NAME", "whisper"))

# aoai_client = AzureOpenAI(
#     azure_deployment=aoai_model_name,
#     api_version=aoai_apiversion,
#     azure_endpoint=aoai_endpoint,
#     api_key=aoai_apikey
# )

# # Configuration of Whisper
# # whisper_endpoint = os.environ["WHISPER_ENDPOINT"]
# # whisper_apikey = os.environ["WHISPER_API_KEY"]
# # whisper_apiversion = os.environ["WHISPER_API_VERSION"]
# # whisper_model_name = os.environ["WHISPER_DEPLOYMENT_NAME"]
# # Create AOAI client for whisper
# whisper_client = AzureOpenAI(
#     api_version=whisper_apiversion,
#     azure_endpoint=whisper_endpoint,
#     api_key=whisper_apikey
# )

# # Function to encode a local video into frames
# def process_video(video_path, frames_per_second=DEFAULT_FRAMES_PER_SECOND, resize=RESIZE_OF_FRAMES, output_dir='', temperature=DEFAULT_TEMPERATURE):
#     print(f"Starting video processing for {video_path} with frames_per_second={frames_per_second}, resize={resize}")
#     base64Frames = []

#     # Prepare the video analysis
#     video = cv2.VideoCapture(video_path)
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = video.get(cv2.CAP_PROP_FPS)
#     frames_to_skip = int(fps / frames_per_second)
#     curr_frame = 0

#     # Prepare to write the frames to disk
#     if output_dir != '':
#         os.makedirs(output_dir, exist_ok=True)
#         frame_count = 1

#     # Loop through the video and extract frames at the specified sampling rate
#     while curr_frame < total_frames - 1:
#         video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
#         success, frame = video.read()
#         if not success:
#             break

#         print(f"Processing frame {curr_frame}/{total_frames}")

#         # Resize the frame if required
#         if resize != 0:
#             height, width, _ = frame.shape
#             frame = cv2.resize(frame, (width // resize, height // resize))

#         _, buffer = cv2.imencode(".jpg", frame)

#         # Save frame as JPG file if output_dir is specified
#         if output_dir != '':
#             frame_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg")
#             with open(frame_filename, "wb") as f:
#                 f.write(buffer)
#             frame_count += 1

#         base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
#         curr_frame += frames_to_skip
#     video.release()
#     print(f"Extracted {len(base64Frames)} frames from {video_path}")

#     return base64Frames

# # Function to transcript the audio from the local video with Whisper
# def process_audio(video_path):
#     print(f"Starting audio transcription for {video_path}")
#     transcription_text = ''
#     try:
#         base_video_path, _ = os.path.splitext(video_path)
#         audio_path = f"{base_video_path}.mp3"
#         clip = VideoFileClip(video_path)
#         clip.audio.write_audiofile(audio_path, bitrate="32k")
#         clip.audio.close()
#         clip.close()
#         print(f"Extracted audio to {audio_path}")

#         # Transcribe the audio
#         print(f"Transcribing audio from {audio_path}")
#         transcription = whisper_client.audio.transcriptions.create(
#             model=whisper_model_name,
#             file=open(audio_path, "rb"),
#         )
#         transcription_text = transcription.text
#         print("Transcript: ", transcription_text + "\n\n")
#     except Exception as ex:
#         print(f'ERROR: {ex}')
#         transcription_text = ''

#     return transcription_text

# # Function to analyze the video with GPT-4.1
# def analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature):
#     print(f"Starting video analysis with system_prompt={system_prompt} and user_prompt={user_prompt}")
#     print(f"Number of frames to analyze: {len(base64frames)}")
#     if transcription:
#         print(f"Including audio transcription in the analysis")

#     try:
#         if transcription: # Include the audio transcription
#             response = aoai_client.chat.completions.create(
#                 model=aoai_model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                     {"role": "user", "content": [
#                         *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
#                         {"type": "text", "text": f"The audio transcription is: {transcription if isinstance(transcription, str) else transcription.text}"}
#                     ]}
#                 ],
#                 temperature=temperature,
#                 max_tokens=4096
#             )
#         else: # Without the audio transcription
#             response = aoai_client.chat.completions.create(
#                 model=aoai_model_name,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                     {"role": "user", "content": [
#                         *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
#                     ]}
#                 ],
#                 temperature=0.5,
#                 max_tokens=4096
#             )

#         json_response = json.loads(response.model_dump_json())
#         response = json_response['choices'][0]['message']['content']
#         print("Analysis completed successfully")

#     except Exception as ex:
#         print(f'ERROR: {ex}')
#         response = f'ERROR: {ex}'

#     return response

# # Split the video into shots of N seconds
# def split_video(video_path, output_dir, shot_interval=DEFAULT_SHOT_INTERVAL, max_duration=None):
#     print(f"Starting video splitting for {video_path} with shot_interval={shot_interval}, max_duration={max_duration}")
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = total_frames / fps

#     if max_duration is not None and max_duration > 0:
#         duration = min(duration, max_duration)

#     for start_time in range(0, int(duration), shot_interval):
#         end_time = min(start_time + shot_interval, duration)
#         output_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(video_path))[0]}_shot_{start_time}-{end_time}_secs.mp4')
#         print(f"Extracting shot from {start_time} to {end_time} into {output_file}")
#         ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)
#         yield output_file

# # Process the video
# def execute_video_processing(st, shot_path, system_prompt, user_prompt, temperature, frames_per_second, analysis_dir):
#     print(f"Starting video processing for shot {shot_path}")
#     # Show the video on the screen
#     st.write(f"Video: {shot_path}:")
#     st.video(shot_path)

#     with st.spinner(f"Analyzing video shot: {shot_path}"):
#         # Extract frames at the specified rate
#         with st.spinner(f"Extracting frames..."):
#             start_time = time.time()
#             if save_frames:
#                 output_dir = os.path.join(analysis_dir, 'frames')
#             else:
#                 output_dir = ''
#             print(f"Extracting frames from {shot_path}")
#             base64frames = process_video(shot_path, frames_per_second=frames_per_second, resize=resize, output_dir=output_dir, temperature=temperature)
#             end_time = time.time()
#             print(f'\t>>>> Frames extraction took {(end_time - start_time):.3f} seconds <<<<')

#         # Extract the transcription of the audio
#         if audio_transcription:
#             print(f"Transcribing audio from {shot_path}")
#             msg = f'Analyzing frames and audio with {aoai_model_name}...'
#             with st.spinner(f"Transcribing audio from video file..."):
#                 start_time = time.time()
#                 transcription = process_audio(shot_path)
#                 end_time = time.time()
#             print(f'Transcription: [{transcription}]')
#             if show_transcription:
#                 st.markdown(f"**Transcription**: {transcription}", unsafe_allow_html=True)
#             print(f'\t>>>> Audio transcription took {(end_time - start_time):.3f} seconds <<<<')
#         else:
#             print(f"Skipping audio transcription")
#             msg = f'Analyzing frames with {aoai_model_name}...'
#             transcription = ''
#         # Analyze the video frames and the audio transcription with GPT-4o
#         with st.spinner(msg):
#             print(f"Analyzing frames with {aoai_model_name}")
#             start_time = time.time()
#             analysis = analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature)
#             end_time = time.time()
#         print(f'\t>>>> Analysis with {aoai_model_name} took {(end_time - start_time):.3f} seconds <<<<')

#     st.success("Analysis completed.")
#     print(f"Analysis completed for shot {shot_path}")
    
#     # Print the analysis content
#     print(f"Analysis content: {analysis}")
    
#     # Save the analysis to a JSON file in the analysis directory
#     analysis_filename = os.path.join(analysis_dir, os.path.splitext(os.path.basename(shot_path))[0] + "_analysis.json")
#     with open(analysis_filename, 'w') as json_file:
#         json.dump({"analysis": analysis}, json_file, indent=4)
#     print(f"Analysis saved as: {analysis_filename}")

#     return analysis

# # # Streamlit User Interface
# # st.set_page_config(
# #     page_title="Video Analysis with GPT-4.1",
# #     layout="centered",
# #     initial_sidebar_state="auto",
# # )
# # st.image("microsoft.png", width=100)
# # st.title('Video Analysis with GPT-4.1')

# with st.sidebar:
#     file_or_url = st.selectbox("Video source:", ["File", "URL"], index=0, help="Select the source, file or url")
#     initial_split = 0

#     if file_or_url == "URL":
#         continuous_transmission = st.checkbox('Continuous transmission', False, help="Video of a continuous transmission")
#         if continuous_transmission:
#             initial_split = DEFAULT_SHOT_INTERVAL
        
#     audio_transcription = st.checkbox('Transcribe audio', True, help="Extract the audio transcription and use in the analysis or not")
#     if audio_transcription:
#         show_transcription = st.checkbox('Show audio transcription', True, help="Present the audio transcription or not")
#     shot_interval = st.number_input(label='Shot interval in seconds', min_value=0, value=DEFAULT_SHOT_INTERVAL, help="The video will be processed in shots based on the number of seconds specified in this field.")
#     frames_per_second = st.number_input('Frames per second', DEFAULT_FRAMES_PER_SECOND, help="The number of frames to extract per second.")
#     resize = st.number_input("Frames resizing ratio", min_value=0, value=RESIZE_OF_FRAMES, help="The size of the images will be reduced in proportion to this number while maintaining the height/width ratio. This reduction is useful for improving latency and reducing token consumption (0 to not resize)")
#     save_frames = st.checkbox('Save the frames to the folder "frames"', True)
#     temperature = float(st.number_input('Temperature for the model', DEFAULT_TEMPERATURE))
#     system_prompt = st.text_area('System Prompt', system_prompt)
#     user_prompt = st.text_area('User Prompt', USER_PROMPT)
#     max_duration = st.number_input('Maximum duration to process (seconds)', 0, help="Specify the maximum duration of the video to process. If the video is longer, only this duration will be processed. Set to 0 to process the entire video.")

# # Video file or Video URL
# if file_or_url == 'File':
#     video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
# else:
#     url = st.text_area("Enter the URL:", value='https://www.youtube.com/watch?v=Y6kHpAeIr4c', height=10)

# # Analyze the video when the button is pressed
# # if st.button("Analyze video", use_container_width=True, type='primary'):

# #     # Show parameters:
# #     print(f"PARAMETERS:")
# #     print(f"file_or_url: {file_or_url}, audio_transcription: {audio_transcription}, shot interval: {shot_interval}, frames per second: {frames_per_second}")
# #     print(f"resize ratio: {resize}, save_frames: {save_frames}, temperature: {temperature}, max_duration: {max_duration}")

# #     if file_or_url == 'URL': # Process Youtube video
# #         st.write(f'Analyzing video from URL {url}...')
        
# #         ydl_opts = {
# #                 'format': '(bestvideo[vcodec^=av01]/bestvideo[vcodec^=vp9]/bestvideo)+bestaudio/best',
# #                 'outtmpl': 'full_video.%(ext)s',
# #                 'force_keyframes_at_cuts': True
# #         }
# #         ydl = yt_dlp.YoutubeDL(ydl_opts)
# #         info_dict = ydl.extract_info(url, download=False)
# #         video_title = info_dict.get('title', 'video')
# #         video_duration = int(info_dict.get('duration', 0))  # Convert to int

# #         # Create a directory for the video analysis
# #         analysis_dir = f"{video_title}_video_analysis"
# #         os.makedirs(analysis_dir, exist_ok=True)

# #         # Create subdirectories for shots and analysis
# #         shots_dir = os.path.join(analysis_dir, "shots")
# #         os.makedirs(shots_dir, exist_ok=True)
# #         analysis_subdir = os.path.join(analysis_dir, "analysis")
# #         os.makedirs(analysis_subdir, exist_ok=True)

# #         # Download the video if it doesn't already exist
# #         video_path = os.path.join(analysis_dir, f"{video_title}.mp4")
# #         if not os.path.exists(video_path):
# #             with st.spinner(f"Downloading video..."):
# #                 ydl_opts['outtmpl'] = video_path
# #                 ydl.download([url])
# #                 print(f"Downloaded video: {video_path}")

# #         if max_duration > 0:
# #             video_duration = min(video_duration, max_duration)
# #         else:
# #             video_duration = int(info_dict.get('duration', 0))  # Convert to int

# #         if shot_interval == 0:
# #             segment_duration = video_duration
# #         else:
# #             segment_duration = int(shot_interval)  # Convert to int

# #         for start in range(0, video_duration, segment_duration):
# #             end = start + segment_duration
# #             shot_filename = f'shot_{start}-{end}.mp4'
# #             shot_path = os.path.join(shots_dir, shot_filename)
# #             with st.spinner(f"Extracting shot from second {start} to {end}..."):
# #                 ffmpeg_extract_subclip(video_path, start, end, targetname=shot_path)
# #                 print(f"Extracted shot: {shot_path}")

# #             # Process the video shot
# #             analysis = execute_video_processing(st, shot_path, system_prompt, user_prompt, temperature, frames_per_second, analysis_subdir)
# #             st.markdown(f"**Description**: {analysis}", unsafe_allow_html=True)

# #             # Example detecting an event
# #             event="electric guitar"
# #             if event in analysis:
# #                 st.write(f'**Detected event "{event}" in shot {shot_path}**')

# #     else: # Process the video file
# #         if video_file is not None:
# #             video_title = os.path.splitext(video_file.name)[0]
# #             analysis_dir = f"{video_title}_video_analysis"
# #             os.makedirs(analysis_dir, exist_ok=True)

# #             # Create subdirectories for shots and analysis
# #             shots_dir = os.path.join(analysis_dir, "shots")
# #             os.makedirs(shots_dir, exist_ok=True)
# #             analysis_subdir = os.path.join(analysis_dir, "analysis")
# #             os.makedirs(analysis_subdir, exist_ok=True)

# #             video_path = os.path.join(analysis_dir, video_file.name)
# #             try:
# #                 with open(video_path, "wb") as f:
# #                     f.write(video_file.getbuffer())
# #                 print(f"Uploaded video file: {video_path}")

# #                 # Splitting video into shots
# #                 for shot_path in split_video(video_path, shots_dir, shot_interval, max_duration):
# #                     print(f"Processing shot: {shot_path}")
# #                     # Process the video shot
# #                     analysis = execute_video_processing(st, shot_path, system_prompt, user_prompt, temperature, frames_per_second, analysis_subdir)
# #                     st.markdown(f"**Description**: {analysis}", unsafe_allow_html=True)

# #             except Exception as ex:
# #                 print(f'ERROR: {ex}')
# #                 st.write(f'ERROR: {ex}')

# if st.button("Analyze video", use_container_width=True, type='primary'):

#     print(f"PARAMETERS:")
#     print(f"file_or_url: {file_or_url}, audio_transcription: {audio_transcription}, shot interval: {shot_interval}, frames per second: {frames_per_second}")
#     print(f"resize ratio: {resize}, save_frames: {save_frames}, temperature: {temperature}, max_duration: {max_duration}")

#     shot_analyses = []  # <-- Store shot analyses here

#     if file_or_url == 'URL':
#         st.write(f'Analyzing video from URL {url}...')
        
#         ydl_opts = {
#                 'format': '(bestvideo[vcodec^=av01]/bestvideo[vcodec^=vp9]/bestvideo)+bestaudio/best',
#                 'outtmpl': 'full_video.%(ext)s',
#                 'force_keyframes_at_cuts': True
#         }
#         ydl = yt_dlp.YoutubeDL(ydl_opts)
#         info_dict = ydl.extract_info(url, download=False)
#         video_title = info_dict.get('title', 'video')
#         video_duration = int(info_dict.get('duration', 0))

#         analysis_dir = f"{video_title}_video_analysis"
#         os.makedirs(analysis_dir, exist_ok=True)
#         shots_dir = os.path.join(analysis_dir, "shots")
#         os.makedirs(shots_dir, exist_ok=True)
#         analysis_subdir = os.path.join(analysis_dir, "analysis")
#         os.makedirs(analysis_subdir, exist_ok=True)

#         video_path = os.path.join(analysis_dir, f"{video_title}.mp4")
#         if not os.path.exists(video_path):
#             with st.spinner(f"Downloading video..."):
#                 ydl_opts['outtmpl'] = video_path
#                 ydl.download([url])
#                 print(f"Downloaded video: {video_path}")

#         if max_duration > 0:
#             video_duration = min(video_duration, max_duration)
#         else:
#             video_duration = int(info_dict.get('duration', 0))

#         if shot_interval == 0:
#             segment_duration = video_duration
#         else:
#             segment_duration = int(shot_interval)

#         for start in range(0, video_duration, segment_duration):
#             end = start + segment_duration
#             shot_filename = f'shot_{start}-{end}.mp4'
#             shot_path = os.path.join(shots_dir, shot_filename)
#             with st.spinner(f"Extracting shot from second {start} to {end}..."):
#                 ffmpeg_extract_subclip(video_path, start, end, targetname=shot_path)
#                 print(f"Extracted shot: {shot_path}")

#             # Store each analysis in a list (don't display yet)
#             analysis = execute_video_processing(
#                 st, shot_path, system_prompt, user_prompt, temperature, frames_per_second, analysis_subdir
#             )
#             shot_analyses.append(analysis)

#     else:  # Process uploaded file
#         if video_file is not None:
#             video_title = os.path.splitext(video_file.name)[0]
#             analysis_dir = f"{video_title}_video_analysis"
#             os.makedirs(analysis_dir, exist_ok=True)
#             shots_dir = os.path.join(analysis_dir, "shots")
#             os.makedirs(shots_dir, exist_ok=True)
#             analysis_subdir = os.path.join(analysis_dir, "analysis")
#             os.makedirs(analysis_subdir, exist_ok=True)
#             video_path = os.path.join(analysis_dir, video_file.name)
#             try:
#                 with open(video_path, "wb") as f:
#                     f.write(video_file.getbuffer())
#                 print(f"Uploaded video file: {video_path}")

#                 # Split and analyze, saving all shot analyses to the list
#                 for shot_path in split_video(video_path, shots_dir, shot_interval, max_duration):
#                     print(f"Processing shot: {shot_path}")
#                     analysis = execute_video_processing(
#                         st, shot_path, system_prompt, user_prompt, temperature, frames_per_second, analysis_subdir
#                     )
#                     shot_analyses.append(analysis)

#             except Exception as ex:
#                 print(f'ERROR: {ex}')
#                 st.write(f'ERROR: {ex}')

#     # ---- After all shots are processed: EXECUTIVE SUMMARY ----
#     if shot_analyses:
#         combined_analyses = "\n\n".join(shot_analyses)
#         summary_prompt = (
#             "You are an expert NFL scouting assistant. Here are detailed play-by-play or segment-by-segment analyses of a player's film. "
#             "Please provide a single, concise executive summary that synthesizes the most important insights, strengths, weaknesses, and player projection, "
#             "removing any redundant details. Write as you would in an NFL scouting report."
#         )

#         with st.spinner("Generating executive summary..."):
#             final_summary = aoai_client.chat.completions.create(
#                 model=aoai_model_name,
#                 messages=[
#                     {"role": "system", "content": summary_prompt},
#                     {"role": "user", "content": combined_analyses}
#                 ],
#                 temperature=temperature,
#                 max_tokens=2048
#             )
#             final_summary_text = json.loads(final_summary.model_dump_json())['choices'][0]['message']['content']

#         st.markdown("## Executive Summary", unsafe_allow_html=True)
#         st.markdown(final_summary_text, unsafe_allow_html=True)

#         # (Optional) Download full details as JSON
#         st.download_button(
#             "Download detailed shot-by-shot analyses (JSON)",
#             data=json.dumps({"analyses": shot_analyses}, indent=2),
#             file_name="detailed_shot_analyses.json"
#         )


import streamlit as st
import cv2
import os
import json
import time
import tempfile
=======
import streamlit as st
import cv2
import os
import time
import json
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
from dotenv import load_dotenv
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from openai import AzureOpenAI
import base64
import uuid

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType
)
from azure.core.credentials import AzureKeyCredential

<<<<<<< HEAD
import yt_dlp

=======
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
# ========== CONFIGURATION ==========
load_dotenv(override=True)
DEFAULT_SHOT_INTERVAL = 30
DEFAULT_FRAMES_PER_SECOND = 1
DEFAULT_TEMPERATURE = 0.5
RESIZE_OF_FRAMES = 4

SYSTEM_PROMPT = '''You are an expert NFL scouting assistant. You will be shown a series of video frames (and transcriptions if available)
from a football game or practice. Analyze and describe the player's physical attributes, athleticism, technique, positional skills,
decision-making, and any notable plays. Highlight strengths, weaknesses, and potential fit for specific NFL roles.
Be objective, concise, and use scouting terminology. If possible, compare the player's traits to current or former NFL athletes.'''
USER_PROMPT = "These are the frames from the video. Focus your analysis on player evaluation and scouting insights."

# ========== STREAMLIT CONFIG ==========
st.set_page_config(page_title="NFL Video Analysis", layout="centered")
st.image("microsoft.png", width=100)
st.title('NFL Video Analysis & Scouting Report')

# ========== TABS ==========
tab1, tab2 = st.tabs(["Video Analysis", "Index & Search"])

# ========== TAB 1: VIDEO ANALYSIS ==========
with tab1:
    with st.sidebar:
        st.header("OpenAI Config")
        aoai_endpoint = st.text_input("Azure OpenAI Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        aoai_apikey = st.text_input("Azure OpenAI API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
        aoai_apiversion = st.text_input("Azure OpenAI API Version", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"))
        aoai_model_name = st.text_input("AOAI Model Name/Deployment", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"))

        st.header("Whisper Speech-to-Text Config")
        whisper_endpoint = st.text_input("Whisper Endpoint", value=os.getenv("WHISPER_ENDPOINT", ""))
        whisper_apikey = st.text_input("Whisper API Key", value=os.getenv("WHISPER_API_KEY", ""), type="password")
        whisper_apiversion = st.text_input("Whisper API Version", value=os.getenv("WHISPER_API_VERSION", "2024-05-01-preview"))
        whisper_model_name = st.text_input("Whisper Deployment Name", value=os.getenv("WHISPER_DEPLOYMENT_NAME", "whisper"))

<<<<<<< HEAD
    file_or_url = st.selectbox("Video source:", ["File", "YouTube URL"], index=0)
    audio_transcription = st.checkbox('Transcribe audio', False)
    shot_interval = st.number_input('Shot interval in seconds', min_value=0, value=DEFAULT_SHOT_INTERVAL)
    frames_per_second = st.number_input('Frames per second', DEFAULT_FRAMES_PER_SECOND)
    resize = st.number_input("Frames resizing ratio", min_value=0, value=RESIZE_OF_FRAMES)
=======
    file_or_url = st.selectbox("Video source:", ["File"], index=0)
    audio_transcription = st.checkbox('Transcribe audio', True)
    shot_interval = st.number_input('Shot interval in seconds', min_value=0, value=DEFAULT_SHOT_INTERVAL)
    frames_per_second = st.number_input('Frames per second', DEFAULT_FRAMES_PER_SECOND)
    resize = st.number_input("Frames resizing ratio", min_value=0, value=RESIZE_OF_FRAMES)
    save_frames = st.checkbox('Save the frames to the folder "frames"', True)
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
    temperature = float(st.number_input('Temperature for the model', DEFAULT_TEMPERATURE))
    system_prompt = st.text_area('System Prompt', SYSTEM_PROMPT)
    user_prompt = st.text_area('User Prompt', USER_PROMPT)
    max_duration = st.number_input('Maximum duration to process (seconds)', 0)

<<<<<<< HEAD
    video_file = None
    video_path = None
    uploaded_or_downloaded = False
    temp_files_to_cleanup = []

    if file_or_url == "File":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    else:
        yt_url = st.text_input("Paste YouTube URL here:", "")
        if yt_url:
            if st.button("Download Video"):
                with st.spinner("Downloading YouTube video..."):
                    ydl_opts = {
                        'format': 'mp4/bestvideo+bestaudio/best',
                        'outtmpl': os.path.join(tempfile.gettempdir(), 'yt_downloaded_video.%(ext)s'),
                        'noplaylist': True,
                        'quiet': True,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(yt_url, download=True)
                        ext = info.get("ext", "mp4")
                        video_path = os.path.join(tempfile.gettempdir(), f'yt_downloaded_video.{ext}')
                        uploaded_or_downloaded = True
                        temp_files_to_cleanup.append(video_path)
                        st.success(f"Downloaded: {info['title']}")
                        st.session_state['yt_downloaded_path'] = video_path
                        st.session_state['yt_title'] = info['title']
        if 'yt_downloaded_path' in st.session_state:
            video_path = st.session_state['yt_downloaded_path']
            uploaded_or_downloaded = True
=======
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645

    # ========== OPENAI CLIENTS ==========
    aoai_client = AzureOpenAI(
        azure_deployment=aoai_model_name,
        api_version=aoai_apiversion,
        azure_endpoint=aoai_endpoint,
        api_key=aoai_apikey
    )
    whisper_client = AzureOpenAI(
        api_version=whisper_apiversion,
        azure_endpoint=whisper_endpoint,
        api_key=whisper_apikey
    )

    # ========== FUNCTIONS ==========
    def process_video(video_path, frames_per_second, resize, output_dir='', temperature=DEFAULT_TEMPERATURE):
        base64Frames = []
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = int(fps / frames_per_second)
        curr_frame = 0
<<<<<<< HEAD
=======
        if output_dir != '':
            os.makedirs(output_dir, exist_ok=True)
            frame_count = 1
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success: break
            if resize != 0:
                height, width, _ = frame.shape
                frame = cv2.resize(frame, (width // resize, height // resize))
            _, buffer = cv2.imencode(".jpg", frame)
<<<<<<< HEAD
=======
            if output_dir != '':
                frame_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg")
                with open(frame_filename, "wb") as f:
                    f.write(buffer)
                frame_count += 1
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            curr_frame += frames_to_skip
        video.release()
        return base64Frames

    def process_audio(video_path):
        transcription_text = ''
        try:
            base_video_path, _ = os.path.splitext(video_path)
            audio_path = f"{base_video_path}.mp3"
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, bitrate="32k")
            clip.audio.close()
            clip.close()
            transcription = whisper_client.audio.transcriptions.create(
                model=whisper_model_name,
                file=open(audio_path, "rb"),
            )
            transcription_text = transcription.text
<<<<<<< HEAD
            # Clean up temp audio
            os.remove(audio_path)
=======
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
        except Exception as ex:
            print(f'ERROR: {ex}')
            transcription_text = ''
        return transcription_text

    def analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature):
        try:
            if transcription:
                response = aoai_client.chat.completions.create(
                    model=aoai_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "user", "content": [
                            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
                            {"type": "text", "text": f"The audio transcription is: {transcription if isinstance(transcription, str) else transcription.text}"}
                        ]}
                    ],
                    temperature=temperature,
                    max_tokens=4096
                )
            else:
                response = aoai_client.chat.completions.create(
                    model=aoai_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "user", "content": [
                            *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
                        ]}
                    ],
                    temperature=temperature,
                    max_tokens=4096
                )
            json_response = json.loads(response.model_dump_json())
            response = json_response['choices'][0]['message']['content']
        except Exception as ex:
            print(f'ERROR: {ex}')
            response = f'ERROR: {ex}'
        return response

    def split_video(video_path, output_dir, shot_interval=DEFAULT_SHOT_INTERVAL, max_duration=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        if max_duration is not None and max_duration > 0:
            duration = min(duration, max_duration)
        for start_time in range(0, int(duration), shot_interval):
            end_time = min(start_time + shot_interval, duration)
<<<<<<< HEAD
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=output_dir) as tmp_shot:
                output_file = tmp_shot.name
=======
            output_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(video_path))[0]}_shot_{start_time}-{end_time}_secs.mp4')
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
            ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)
            yield output_file

    def get_total_shots(video_path, shot_interval, max_duration=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        if max_duration and max_duration > 0:
            duration = min(duration, max_duration)
        return int((duration + shot_interval - 1) // shot_interval)

    def generate_summary(analyses, openai_deployment, openai_api_version, openai_endpoint, openai_apikey):
        if not analyses:
            return "No analyses to summarize."
        summary = "\n\n---\n\n".join(analyses)
        prompt = (
            "You are a helpful NFL scouting assistant. Based on the following scouting reports, summarize into one concise report.\n\n"
            f"Scouting Reports:\n{summary}\n\nSummary:"
        )
        sum_openai_client = AzureOpenAI(
            azure_deployment=openai_deployment,
            api_version=openai_api_version,
            azure_endpoint=openai_endpoint,
            api_key=openai_apikey
        )
        response = sum_openai_client.chat.completions.create(
            model=openai_deployment,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        json_response = json.loads(response.model_dump_json())
        return json_response['choices'][0]['message']['content']

    # ========== ANALYSIS ==========
    if st.button("Analyze video", use_container_width=True, type='primary'):
<<<<<<< HEAD
        if (file_or_url == "File" and video_file is not None) or (file_or_url == "YouTube URL" and video_path is not None):
            # If file uploaded, save to temp. If YT, already have path.
            if file_or_url == "File":
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(video_file.getbuffer())
                    video_path = tmp.name
                temp_files_to_cleanup.append(video_path)

            st.video(video_path)  # Show the video once at top

=======
        if video_file is not None:
            video_title = os.path.splitext(video_file.name)[0]
            analysis_dir = f"{video_title}_video_analysis"
            os.makedirs(analysis_dir, exist_ok=True)
            shots_dir = os.path.join(analysis_dir, "shots")
            os.makedirs(shots_dir, exist_ok=True)

            video_path = os.path.join(analysis_dir, video_file.name)
            with open(video_path, "wb") as f:
                f.write(video_file.getbuffer())

            # Show the video only ONCE at the top!
            st.video(video_file)

            # Progress reporting
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
            total_shots = get_total_shots(video_path, shot_interval, max_duration)
            progress_msg = st.empty()

            all_analyses = []
<<<<<<< HEAD
            temp_shots = []
            for i, shot_path in enumerate(split_video(video_path, tempfile.gettempdir(), shot_interval, max_duration)):
                temp_shots.append(shot_path)
=======
            for i, shot_path in enumerate(split_video(video_path, shots_dir, shot_interval, max_duration)):
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
                base64frames = process_video(
                    shot_path,
                    frames_per_second=frames_per_second,
                    resize=resize,
<<<<<<< HEAD
                    output_dir='',   # Not saving frames!
=======
                    output_dir='frames' if save_frames else '',
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
                    temperature=temperature
                )
                transcription = process_audio(shot_path) if audio_transcription else ''
                analysis = analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature)
                all_analyses.append(analysis)
                progress_msg.info(f"Analysis {i+1}/{total_shots} complete")

<<<<<<< HEAD
            # Clean up temp shot files
            for f in temp_shots:
                try: os.remove(f)
                except: pass
            for f in temp_files_to_cleanup:
                try: os.remove(f)
                except: pass

=======
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
            concise_summary = generate_summary(
                all_analyses,
                openai_deployment=aoai_model_name,
                openai_api_version=aoai_apiversion,
                openai_endpoint=aoai_endpoint,
                openai_apikey=aoai_apikey
            )
            st.session_state['analyses'] = all_analyses
            st.session_state['full_summary'] = concise_summary

            st.markdown("### Final Combined Summary Analysis")
            st.markdown(concise_summary, unsafe_allow_html=True)
            st.success("Analysis complete. Switch to 'Index & Search' tab to index or chat over the results.")
            st.balloons()
<<<<<<< HEAD
        else:
            st.warning("Please upload a video or download a YouTube video first.")
=======
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645

# ========== TAB 2: INDEXING & SEARCH ==========
with tab2:
    st.header("Index & Search (RAG)")

<<<<<<< HEAD
=======
    # Credentials UI (again, can be made global if preferred)
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
    search_service_endpoint = st.text_input("Azure Search Endpoint", value=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", ""))
    search_api_key = st.text_input("Azure Search API Key", value=os.getenv("AZURE_SEARCH_API_KEY", ""), type="password")
    index_name = st.text_input("Index Name", value="nfl-player-scouting")
    openai_endpoint = st.text_input("OpenAI Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    openai_apikey = st.text_input("OpenAI API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
    openai_deployment = st.text_input("OpenAI Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"))
    openai_api_version = st.text_input("OpenAI API Version", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"))

<<<<<<< HEAD
=======
    # Index button
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
    if 'analyses' in st.session_state:
        def extract_player_name_and_position(analysis_text):
            import re
            name_match = re.search(r"Player.*?:\s*([A-Za-z0-9\.\s'#-]+)", analysis_text)
            pos_match = re.search(r"Position.*?:\s*([A-Za-z0-9\s/-]+)", analysis_text)
            player = name_match.group(1).strip() if name_match else "Unknown"
            pos = pos_match.group(1).strip() if pos_match else "Unknown"
            return player, pos

        if st.button("Index Last Analysis Results"):
            raw_analyses = st.session_state['analyses']
            documents = []
            for analysis in raw_analyses:
                player, position = extract_player_name_and_position(analysis)
                documents.append({
                    "id": str(uuid.uuid4()),
                    "player_name": player,
                    "position": position,
                    "content": analysis
                })

            search_client = SearchClient(
                endpoint=search_service_endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(search_api_key)
            )
            index_client = SearchIndexClient(
                endpoint=search_service_endpoint,
                credential=AzureKeyCredential(search_api_key)
            )
            try:
                index_client.get_index(index_name)
            except Exception:
                fields = [
                    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                    SearchableField(name="player_name", type=SearchFieldDataType.String, sortable=True, filterable=True),
                    SearchableField(name="position", type=SearchFieldDataType.String, sortable=True, filterable=True),
                    SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
                ]
                index = SearchIndex(name=index_name, fields=fields)
                index_client.create_index(index)

            search_client.upload_documents(documents)
            st.success(f"Indexed {len(documents)} analyses!")

    else:
        st.info("Run video analysis in Tab 1 first!")

    # Simple search + RAG Q&A
    st.subheader("Search / Q&A Over Analyses")
    query = st.text_input("Ask a question or enter search terms:")
    if query and search_service_endpoint and search_api_key and openai_apikey:
        search_client = SearchClient(
            endpoint=search_service_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_api_key)
        )
        results = list(search_client.search(query))
<<<<<<< HEAD
=======
        # st.write("### Top Results")
        # for r in results[:5]:
        #     st.markdown(f"**Player:** {r['player_name']} ({r['position']})\n\n{r['content'][:500]}...")
        # RAG Q&A
>>>>>>> e94241271f68c57e9de07b2876e4506fc217f645
        top_docs = "\n\n".join([r['content'] for r in results[:3]])
        prompt = (
            "You are a helpful NFL scouting assistant. Based on the following scouting reports, answer the user's question.\n\n"
            f"Scouting Reports:\n{top_docs}\n\nUser Question: {query}\n\nAnswer:"
        )
        openai_client = AzureOpenAI(
            azure_deployment=openai_deployment,
            api_version=openai_api_version,
            azure_endpoint=openai_endpoint,
            api_key=openai_apikey
        )
        response = openai_client.chat.completions.create(
            model=openai_deployment,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        json_response = json.loads(response.model_dump_json())
        rag_answer = json_response['choices'][0]['message']['content']
        st.markdown("### RAG Answer")
        st.markdown(rag_answer)
