from groq import Groq
import edge_tts
import asyncio
import os
import whisper_timestamped as whisper
from dotenv import load_dotenv
from utility.video.video_search_query_generator import getVideoSearchQueriesTimed, merge_empty_intervals
from utility.captions.timed_captions_generator import generate_timed_captions
from utility.script.script_generator import generate_script
from utility.audio.audio_generator import generate_audio
from utility.captions.timed_captions_generator import generate_timed_captions
from utility.video.background_video_generator import generate_video_url
from utility.render.render_engine import get_output_media
from utility.video.video_search_query_generator import getVideoSearchQueriesTimed, merge_empty_intervals


load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. Generate a script from Google NotebookLLM for the given learning topic
# 
script_from_gllm = """
Have you ever followed a recipe.
"""

# response = groq.chat.completions.create(
#     model="llama3-8b-8192",
#     messages=[{"role": "user", "content": script_from_gllm}],
# )

# print(response)

async def generate_audio(text,outputFilename):
    communicate = edge_tts.Communicate(text,"en-AU-WilliamNeural")
    await communicate.save(outputFilename)
SAMPLE_FILE_NAME = "output.wav"
asyncio.run(generate_audio(script_from_gllm, SAMPLE_FILE_NAME))


audio = whisper.load_audio(SAMPLE_FILE_NAME)
model = whisper.load_model("base", device="cpu")
# caption_t = whisper.transcribe(model, audio, language="en", task="transcribe")
caption_t = generate_timed_captions(SAMPLE_FILE_NAME)
# print(result)

search_terms=getVideoSearchQueriesTimed(script=script_from_gllm,captions_timed=caption_t)
# gen = transcribe_timestamped(WHISPER_MODEL, audio_filename, verbose=False, fp16=False)
# timed_captions = generate_timed_captions(SAMPLE_FILE_NAME)
# print(timed_captions)
VIDEO_SERVER = "pexel"
background_video_urls = None
if search_terms is not None:
    background_video_urls = generate_video_url(search_terms, VIDEO_SERVER)
    print(background_video_urls)
else:
    print("No background video")

background_video_urls = merge_empty_intervals(background_video_urls)

if background_video_urls is not None:
    video = get_output_media(SAMPLE_FILE_NAME, caption_t, background_video_urls, VIDEO_SERVER)
    print(video)
else:
    print("No video")