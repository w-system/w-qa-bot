import streamlit as st
from rag import RagOpenAI


st.title('W-SYSTEM CHAT-BOT')


st.caption("🚀 A Streamlit chatbot powered by OpenAI")
st.chat_message("assistant").write("何かご用でしょうか?")

prompt = st.chat_input("Say something")
if prompt:
    st.chat_message("user").write(prompt)
    rag = RagOpenAI(prompt)
    st.chat_message("assistant").write("少々お待ちください")
    answer = rag.call_func_rag()
    st.chat_message("assistant").write(answer)
