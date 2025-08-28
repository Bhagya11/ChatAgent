import streamlit as st
from typing import TypedDict
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

# 🔹 Change base_url to your ngrok URL
OLLAMA_BASE_URL = "https://abcd1234.ngrok-free.app"

class AgentState(TypedDict):
    input: str
    output: str

llm = ChatOllama(model="gemma3:1b", base_url=OLLAMA_BASE_URL)

def chatbot_node(state: AgentState):
    user_message = state["input"]
    response = llm.invoke(user_message)
    return {"output": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)
app_graph = workflow.compile()

st.set_page_config(page_title="LangGraph Ollama Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 LangGraph Agent (Gemma3:1b via Ollama)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Say something...")

if user_input:
    state = {"input": user_input}
    final_state = app_graph.invoke(state)
    bot_reply = final_state["output"]

    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", bot_reply))

for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
