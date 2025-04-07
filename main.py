import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import re

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(api_key= groq_api_key, model="llama3-70b-8192")

st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")
st.title("YouTube Video Summarizer With LLama3")
st.markdown("Paste a YouTube Video URL below to get a summary of the video.")

url = st.text_input("Enter YouTube Video URL")

def extract_video_id(url):
    pattern= r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match[1] if match else None

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None

if url:
    if not (video_id := extract_video_id(url)):
        st.error("Invalid YouTube URL. Please enter a valid one.")
    elif transcript_text := fetch_transcript(video_id):
        prompt = PromptTemplate.from_template("You are an expert summarizer. Summarize this YouTube transcript:\n\n{transcript}")
        chain = LLMChain(llm=llm, prompt=prompt)

        with st.spinner("Summarizing with LLaMA 3...."):
            summary = chain.run(transcript=transcript_text)

        st.subheader("Video Summary")
        st.write(summary)
