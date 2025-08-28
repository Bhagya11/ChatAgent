import streamlit as st
from typing import TypedDict
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END

# --- LangGraph setup ---
class AgentState(TypedDict):
    input: str
    output: str

llm = ChatOllama(model="gemma3:1b")

def chatbot_node(state: AgentState):
    user_message = state["input"]
    response = llm.invoke(user_message)
    return {"output": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.set_entry_point("chatbot")
workflow.add_edge("chatbot", END)
app_graph = workflow.compile()

# --- Streamlit UI ---
st.set_page_config(page_title="LangGraph Ollama Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 LangGraph Agent with Ollama (Gemma3:1b)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Say something...")

if user_input:
    # Run through LangGraph agent
    state = {"input": user_input}
    final_state = app_graph.invoke(state)
    bot_reply = final_state["output"]

    # Save history
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("bot", bot_reply))

# Display chat history
for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)


