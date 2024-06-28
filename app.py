import streamlit as st
from _chain_ import ContentAssistant_run
from _prompts import Content_Assistant
from create_datavector import retrieve_knowledge, text_load, get_chunk, vector_data

# Tải nội dung từ tệp tin
files = ['quy_trinh_viet_content.txt']
documents = text_load(files)
text_chunks = get_chunk(documents)
vector_store = vector_data(text_chunks)
file_txt_content = ' '.join([doc.page_content for doc in documents])
print("Vector store created and documents loaded.")# Nối nội dung của các tài liệu lại

st.title("Content Assistant Bot")
st.write("Chào mừng bạn đến với Content Assistant Bot")

bot = ContentAssistant_run()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_input = st.text_input("Bạn: ", key="user_input")

if st.button("Gửi"):
    collection_name = 'documents'
    query = user_input

    file_txt_content = retrieve_knowledge(query, collection_name, file_txt_content)

    bot_response = bot.collect_messages(
        user_input,
        Content_Assistant.CALL_SUPPORT_PROMPT.format(file_txt=file_txt_content),
        Content_Assistant.ROUTER_PROMPT.format(file_txt=file_txt_content),
        Content_Assistant.TECH_PROMPT.format(file_txt=file_txt_content),
        Content_Assistant.METHOD_PROMPT.format(file_txt=file_txt_content),
        Content_Assistant.CONTEXT_PROMPT.format(file_txt=file_txt_content)
    )

    st.session_state['messages'].append({"role": "user", "content": user_input})
    st.session_state['messages'].append({"role": "assistant", "content": bot_response})

    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.write(f"**Bạn:** {message['content']}")
        else:
            st.write(f"**Assistant:** {message['content']}")
