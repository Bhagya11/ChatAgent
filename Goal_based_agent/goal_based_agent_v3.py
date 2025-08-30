import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import re

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyAJoievCdhnH4VUJjTVZ-Vkp1J3v1D53ao")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found! Add it to your .env file.")
    st.stop()

# Initialize Gemini 2.5 Pro
llm = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Store application data
application_info = {"name": None, "email": None, "skills": None}

# Functions for tools
def extract_application_info(text: str) -> str:
    name_match = re.search(r"(?:my name is|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text, re.IGNORECASE)
    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
    skills_match = re.search(r"(?:skills are|i know|i can use)\s+(.+)", text, re.IGNORECASE)

    response = []

    if name_match:
        application_info["name"] = name_match.group(1).title()
        response.append("✅ Name saved.")
    if email_match:
        application_info["email"] = email_match.group(0)
        response.append("✅ Email saved.")
    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()
        response.append("✅ Skills saved.")

    if not any([name_match, email_match, skills_match]):
        return "❓ I couldn't extract any info. Could you please provide your name, email, or skills?"

    return " ".join(response) + " Let me check what else I need."

def check_application_goal(_: str) -> str:
    if all(application_info.values()):
        return f"✅ You're ready! Name: {application_info['name']}, Email: {application_info['email']}, Skills: {application_info['skills']}."
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f"⏳ Still need: {', '.join(missing)}. Please provide this."

# Tools
tools = [
    Tool(
        name="extract_application_info",
        func=extract_application_info,
        description="Extracts name, email, and skills from user input."
    ),
    Tool(
        name="check_application_goal",
        func=check_application_goal,
        description="Checks if all details are collected. Tells user what is missing.",
        return_direct=True
    )
]

# System prompt
SYSTEM_PROMPT = """You are a helpful job application assistant.
Your goal is to collect the user's name, email, and skills.
Once everything is collected, confirm that the application is complete.
"""

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    agent_kwargs={"system_message": SYSTEM_PROMPT}
)

# Streamlit UI
st.set_page_config(page_title="Job Application Assistant", page_icon="🤖", layout="centered")
st.title("🤖 Job Application Assistant")
st.write("Hi! Please tell me your **name**, **email**, and **skills**.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get agent response
    response = agent.invoke({"input": user_input})
    bot_reply = response["output"]

    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    # If application is complete
    if "you're ready" in bot_reply.lower():
        st.success("🎉 Application info complete!")
