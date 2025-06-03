import streamlit as st
import cv2
import os
import time
import json
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

    file_or_url = st.selectbox("Video source:", ["File"], index=0)
    audio_transcription = st.checkbox('Transcribe audio', True)
    shot_interval = st.number_input('Shot interval in seconds', min_value=0, value=DEFAULT_SHOT_INTERVAL)
    frames_per_second = st.number_input('Frames per second', DEFAULT_FRAMES_PER_SECOND)
    resize = st.number_input("Frames resizing ratio", min_value=0, value=RESIZE_OF_FRAMES)
    save_frames = st.checkbox('Save the frames to the folder "frames"', True)
    temperature = float(st.number_input('Temperature for the model', DEFAULT_TEMPERATURE))
    system_prompt = st.text_area('System Prompt', SYSTEM_PROMPT)
    user_prompt = st.text_area('User Prompt', USER_PROMPT)
    max_duration = st.number_input('Maximum duration to process (seconds)', 0)

    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

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
        if output_dir != '':
            os.makedirs(output_dir, exist_ok=True)
            frame_count = 1
        while curr_frame < total_frames - 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
            success, frame = video.read()
            if not success: break
            if resize != 0:
                height, width, _ = frame.shape
                frame = cv2.resize(frame, (width // resize, height // resize))
            _, buffer = cv2.imencode(".jpg", frame)
            if output_dir != '':
                frame_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg")
                with open(frame_filename, "wb") as f:
                    f.write(buffer)
                frame_count += 1
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
            output_file = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(video_path))[0]}_shot_{start_time}-{end_time}_secs.mp4')
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
            total_shots = get_total_shots(video_path, shot_interval, max_duration)
            progress_msg = st.empty()

            all_analyses = []
            for i, shot_path in enumerate(split_video(video_path, shots_dir, shot_interval, max_duration)):
                base64frames = process_video(
                    shot_path,
                    frames_per_second=frames_per_second,
                    resize=resize,
                    output_dir='frames' if save_frames else '',
                    temperature=temperature
                )
                transcription = process_audio(shot_path) if audio_transcription else ''
                analysis = analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature)
                all_analyses.append(analysis)
                progress_msg.info(f"Analysis {i+1}/{total_shots} complete")

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

# ========== TAB 2: INDEXING & SEARCH ==========
with tab2:
    st.header("Index & Search (RAG)")

    # Credentials UI (again, can be made global if preferred)
    search_service_endpoint = st.text_input("Azure Search Endpoint", value=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT", ""))
    search_api_key = st.text_input("Azure Search API Key", value=os.getenv("AZURE_SEARCH_API_KEY", ""), type="password")
    index_name = st.text_input("Index Name", value="nfl-player-scouting")
    openai_endpoint = st.text_input("OpenAI Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    openai_apikey = st.text_input("OpenAI API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
    openai_deployment = st.text_input("OpenAI Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"))
    openai_api_version = st.text_input("OpenAI API Version", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"))

    # Index button
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
        # st.write("### Top Results")
        # for r in results[:5]:
        #     st.markdown(f"**Player:** {r['player_name']} ({r['position']})\n\n{r['content'][:500]}...")
        # RAG Q&A
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
