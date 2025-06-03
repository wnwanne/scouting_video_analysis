import os
import yt_dlp
from yt_dlp.utils import download_range_func
from moviepy.editor import VideoFileClip
import time

def convert_to_mp4(input_path, output_path, target_size_mb=200):
    # Load the video file
    clip = VideoFileClip(input_path)
    
    # Calculate the bitrate to keep the file under the target size
    duration = clip.duration
    target_bitrate = (target_size_mb * 8 * 1024 * 1024) / duration  # in bits per second
    
    # Reduce the bitrate further to ensure the file size is under 200 MB
    target_bitrate = target_bitrate * 0.6  # Reduce by 40%
    
    # Write the video file to MP4 format with the calculated bitrate
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac', bitrate=f'{int(target_bitrate)}')
    
    # Close the clip
    clip.close()

def main():
    # Ask for YouTube URL
    url = input("Enter the YouTube URL: ")

    # Ask for start time in seconds (default 0)
    start_str = input("Enter start time in seconds (default 0): ") or "0"
    start_time = int(start_str)

    # Ask for end time in seconds (default 60)
    end_str = input("Enter end time in seconds (default 60): ") or "60"
    end_time = int(end_str)

    # Ask for output directory (default 'output')
    output_dir = input("Enter the output directory (default 'output'): ") or "output"
    os.makedirs(output_dir, exist_ok=True)

    # Download options
    ydl_opts = {
        'format': '(bestvideo[vcodec^=av01]/bestvideo[vcodec^=vp9]/bestvideo)+bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
        'download_ranges': download_range_func(None, [(start_time, end_time)])
    }

    # Download video
    print("Downloading video segment...")
    start_time_download = time.time()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_video = ydl.prepare_filename(info)  # Full path to downloaded video
    end_time_download = time.time()
    download_duration = end_time_download - start_time_download
    print(f"Download completed in {download_duration:.2f} seconds")

    # Extract the title from the info dictionary
    title = info.get('title', 'segment')

    # Name the segment file based on the title and start and end times
    segment_output = os.path.join(output_dir, f"{title}_segment_{start_time}-{end_time}.mp4")
    
    # Convert to MP4 and ensure the file is under 200 MB
    start_time_conversion = time.time()
    convert_to_mp4(downloaded_video, segment_output, target_size_mb=200)
    end_time_conversion = time.time()
    conversion_duration = end_time_conversion - start_time_conversion
    print(f"Conversion completed in {conversion_duration:.2f} seconds")
    
    # # Remove the original downloaded file
    # os.remove(downloaded_video)
    
    print(f"Segment saved as: {segment_output}")

if __name__ == "__main__":
    main()