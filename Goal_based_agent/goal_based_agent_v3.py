import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import re
import fitz  # PyMuPDF

# -----------------------------
# Load API Key
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY","AIzaSyAJoievCdhnH4VUJjTVZ-Vkp1J3v1D53ao")


if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")
    st.stop()

# -----------------------------
# Initialize Gemini LLM
# -----------------------------
llm = GoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Global storage for applicant info
# -----------------------------
application_info = {"name": None, "email": None, "skills": None}

# -----------------------------
# Helper Functions
# -----------------------------

def extract_application_info(text: str) -> str:
    """Extract Name, Email, Skills from user chat text."""
    name_match = re.search(r"(?:my name is|i am|name:)\s*([a-zA-Z][a-zA-Z\s]+)", text, re.IGNORECASE)
    email_match = re.search(r"(?:email:|mail:)?\s*([\w\.-]+@[\w\.-]+\.\w+)", text, re.IGNORECASE)
    skills_match = re.search(r"(?:skills are|i know|i can use|skills:)\s*([a-zA-Z0-9, \-_/]+)", text, re.IGNORECASE)

    if name_match:
        application_info["name"] = name_match.group(1).strip().title()
    if email_match:
        application_info["email"] = email_match.group(1).strip()
    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()

    return "Got it. Let me check what else I need."


def extract_text_from_pdf(uploaded_file):
    """Extracts all text from a PDF file using PyMuPDF."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def extract_info_from_cv(text: str):
    """Extract Name, Email, Skills from raw resume text."""
    extracted_info = {"name": None, "email": None, "skills": None}

    # Regex-based extraction
    name_match = re.search(r"(?:full\s*name:|name:)\s*([a-zA-Z][a-zA-Z\s]+)", text, re.IGNORECASE)
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    skills_match = re.search(r"(?:skills:)\s*([a-zA-Z0-9, \-_/]+)", text, re.IGNORECASE)

    if name_match:
        extracted_info["name"] = name_match.group(1).strip()
    if email_match:
        extracted_info["email"] = email_match.group(0).strip()
    if skills_match:
        extracted_info["skills"] = skills_match.group(1).strip()

    # Fallback: Guess name and skills if labels are missing
    if not extracted_info["name"]:
        name_guess = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        if name_guess:
            extracted_info["name"] = name_guess.group(1).strip()

    if not extracted_info["skills"]:
        skill_keywords = re.findall(r"(Python|Java|C\+\+|Machine Learning|AI|SQL|Excel|TensorFlow|PyTorch)", text, re.IGNORECASE)
        if skill_keywords:
            extracted_info["skills"] = ", ".join(set([s.capitalize() for s in skill_keywords]))

    return extracted_info


def check_application_goal(_: str) -> str:
    """Check if Name, Email, Skills are all collected."""
    if all(application_info.values()):
        return f"✅ You're ready! Name: {application_info['name']}, Email: {application_info['email']}, Skills: {application_info['skills']}."
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f"⏳ Still need: {', '.join(missing)}"


# -----------------------------
# LangChain Agent Tools
# -----------------------------
tools = [
    Tool(name="extract_application_info", func=extract_application_info, description="Extract name, email, and skills"),
    Tool(name="check_application_goal", func=check_application_goal, description="Check if all application info is collected")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False
)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="🎯 Job Application Assistant", layout="centered")
st.title("🤖 Gemini-Powered Job Application Assistant")
st.markdown("Tell me your **name**, **email**, and **skills** to complete your application!")

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "goal_complete" not in st.session_state:
    st.session_state.goal_complete = False
if "download_ready" not in st.session_state:
    st.session_state.download_ready = False
if "application_summary" not in st.session_state:
    st.session_state.application_summary = ""

# -----------------------------
# Sidebar: Resume Upload
# -----------------------------
st.sidebar.header("📤 Upload Resume (Optional)")
resume = st.sidebar.file_uploader("Upload your resume", type=["pdf", "txt"])

if resume:
    st.sidebar.success("Resume uploaded!")
    text = extract_text_from_pdf(resume) if resume.type == "application/pdf" else resume.read().decode("utf-8")
    extracted = extract_info_from_cv(text)
    for key in application_info:
        if extracted[key]:
            application_info[key] = extracted[key]
    st.sidebar.info("🔍 Extracted info from resume:")
    for key, value in extracted.items():
        st.sidebar.markdown(f"**{key.capitalize()}:** {value if value else 'Not found'}")

# -----------------------------
# Sidebar: Reset Button
# -----------------------------
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.chat_history.clear()
    st.session_state.goal_complete = False
    st.session_state.download_ready = False
    st.session_state.application_summary = ""
    for key in application_info:
        application_info[key] = None
    st.experimental_rerun()

# -----------------------------
# Main Chat
# -----------------------------
user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    extract_application_info(user_input)
    response = agent.invoke({"input": user_input})
    bot_reply = response["output"]
    st.session_state.chat_history.append(("bot", bot_reply))
    goal_status = check_application_goal("check")
    st.session_state.chat_history.append(("status", goal_status))

    if "you're ready" in goal_status.lower():
        st.session_state.goal_complete = True
        summary = (
            f"✅ Name: {application_info['name']}\n"
            f"📧 Email: {application_info['email']}\n"
            f"🛠️ Skills: {application_info['skills']}\n"
        )
        st.session_state.application_summary = summary
        st.session_state.download_ready = True

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "user":
        with st.chat_message("🧑"):
            st.markdown(message)
    elif sender == "bot":
        with st.chat_message("🤖"):
            st.markdown(message)
    elif sender == "status":
        with st.chat_message("📊"):
            st.info(message)

# Show completion and download
if st.session_state.goal_complete:
    st.success("🎉 All information collected! You're ready to apply!")

if st.session_state.download_ready:
    st.download_button(
        label="📥 Download Application Summary",
        data=st.session_state.application_summary,
        file_name="application_summary.txt",
        mime="text/plain"
    )
