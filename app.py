import streamlit as st
from rag import RagOpenAI


st.title('W-SYSTEM CHAT-BOT')

prompt = st.chat_input("Say something")
messages = st.container(height=600)
if prompt:
    messages.chat_message("user").write(prompt)
    messages.chat_message("assistant").write("Thinking...")
    rag = RagOpenAI(prompt)
    answer = rag.call_func_rag()
    messages.chat_message("assistant").write(answer)