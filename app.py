import streamlit as st
from streamlit import session_state as ss
from dotenv import load_dotenv
import gtts
from gtts import gTTS
from playsound import playsound
# from final_finetuned_pegasus_model import generate_shortest_summary
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

load_dotenv() ##load all the nevironment variables
import os
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt="""You are Yotube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in paragraph
within 500 words. Please provide the summary of the text given here:  """

## getting the summary based on Prompt from Google Gemini Pro
def generate_gemini_content(transcript_text,prompt):

    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(prompt+transcript_text)
    return response.text

def generate_shortest_summary(detailed_summary):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained("pegasus-samsum-model")
    gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
    pipe = pipeline("summarization", model=model_pegasus,tokenizer=tokenizer)
    model_summary = pipe(detailed_summary, **gen_kwargs)[0]["summary_text"]
    return model_summary

## getting the transcript data from yt videos
def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e
    


st.title("YouTube Transcript to Detailed Notes Converter")
youtube_link = st.text_input("Enter YouTube Video Link:")

if 'b1_count' not in ss:
    ss.b1_count = 0

def count(key):
    ss[key] += 1
if youtube_link:
    video_id = youtube_link.split("=")[1]
    print(video_id)
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

st.button('Get Detailed Notes', key = 'b1', on_click=count, args=('b1_count',))

button1 = bool(ss.b1_count % 2)

if button1 :
    transcript_text=extract_transcript_details(youtube_link)

    if transcript_text:
        summary=generate_gemini_content(transcript_text,prompt)
        st.markdown("## Detailed Notes:")
        st.write(summary)
       
        summarized_summary=generate_shortest_summary(summary)
        st.markdown("## Shortest Summary:")
        st.write(summarized_summary)

        language = 'en'  
        obj = gTTS(text=summarized_summary, lang=language, slow=False) 
        obj.save("result.mp3")
        playsound("result.mp3")
        os.remove("result.mp3")

    
    

