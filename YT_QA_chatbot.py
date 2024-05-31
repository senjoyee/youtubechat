import uuid
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.prebuilt import chat_agent_executor
import streamlit as st
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Function to load and process YouTube transcript
def chat_doc(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(transcript)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    memory = SqliteSaver.from_conn_string(":memory:")
    tool = create_retriever_tool(
        retriever,
        "youtube_transcript_retriever",  # Changed name to a valid pattern
        "Searches and returns excerpts from the youtube transcript.",
    )
    tools = [tool]
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools, checkpointer=memory)
    return agent_executor, memory, vectorstore

# Streamlit UI
st.title("YouTube Transcript Chatbot")

if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
    st.session_state.memory = None
    st.session_state.vectorstore = None

video_url = st.text_input("Enter YouTube Video URL:")
if st.button("Load Video"):
    if video_url:
        st.session_state.agent_executor, st.session_state.memory, st.session_state.vectorstore = chat_doc(video_url)
        st.write("Chatbot is ready! Ask your questions below.")
    else:
        st.write("Please enter a valid YouTube video URL.")

if st.session_state.agent_executor:
    user_input = st.text_input("Your Question:")
    if user_input:
        # Generate a unique thread_id for the session
        thread_id = str(uuid.uuid4())
        # Prepare the input for the agent executor
        input_data = {"messages": [("user", user_input)]}
        # Include the thread_id in the config
        config = {"configurable": {"thread_id": thread_id}}
        response = st.session_state.agent_executor.invoke(input_data, config)
        # Extract the last message from the response
        last_message = response["messages"][-1]
        st.write(last_message.content if isinstance(last_message, AIMessage) else last_message)

if st.button("Clear Memory"):
    st.session_state.agent_executor = None
    st.session_state.memory = None
    st.session_state.vectorstore = None
    st.write("Memory cleared. Please enter a new YouTube video URL.")
