import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
# from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENAI_API_KEY"] = ''

embeddings = OpenAIEmbeddings()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get(''),
)


class ContentAssistant_run:
    def __init__(self):
        self.history = []

    def save_history(self):
        with open('chat_history.json', 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def load_history(self):
        if os.path.exists('chat_history.json'):
            with open('chat_history.json', 'r', encoding='utf-8') as f:
                self.history = json.load(f)

    def intent_detection(self, conversation, router_prompt):
        messages = [
            {'role': 'user', 'content': router_prompt.format(conversation=conversation)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    def tech_prompt(self, conversation, tech_prompt):
        messages = [
            {'role': 'user', 'content': tech_prompt.format(conversation=conversation)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    def method_prompt(self, conversation, method_prompt):
        messages = [
            {'role': 'user', 'content': method_prompt.format(conversation=conversation)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    def context_prompt(self, conversation, context_prompt):
        messages = [
            {'role': 'user', 'content': context_prompt.format(conversation=conversation)}
        ]
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0
        )
        return response.choices[0].message.content.strip(" \n")

    # def save_to_vectorstore(self, user_input, agent_response):
    #     documents = [
    #         {"text": user_input, "metadata": {"role": "user"}},
    #         {"text": agent_response, "metadata": {"role": "assistant"}}
    #     ]
    #     text_chunks = get_chunk(documents)
    #     vector_store = load_vector(text_chunks, "new_assistant_documents")
    #     return vector_store

    def collect_messages(self, user_input, call_support_prompt, router_prompt, tech_prompt, method_prompt, context_prompt):
        self.load_history()
        # file_txt_content = self.retrieve_file_content("quy_trinh_viet_content")
        messages = self.history.copy()
        messages.append(
            {'role': 'user', 'content': call_support_prompt.format(chat_history=self.history, user_input=user_input)}
        )
        response = client.chat.completions.create(
            model='gpt-4',
            messages=messages,
            temperature=0
        )
        agent_response = response.choices[0].message['content']
        self.history.append({'role': 'assistant', 'content': agent_response})
        self.save_history()

        if 'END_OF_CONVERSATION' in agent_response:
            return "Assistant: Cảm ơn bạn đã kết nối với chúng tôi, chúc bạn một ngày tốt lành"

        intent = self.intent_detection(user_input, router_prompt)

        if 'OUT_OF_CONTEXT' in intent:
            agent_response = 'Xin lỗi khách hàng và nói về nhiệm vụ của bạn'
        elif 'GREETING' in intent:
            agent_response = 'Xin chào! Tôi là chatbot của Mekong AI, sẵn sàng hỗ trợ bạn về những vấn đề liên quan đến content như kỹ thuật viết content, phương pháp viết content, hướng dẫn bạn viết bài content. Bạn cần giúp đỡ về vấn đề gì?'
        elif 'TECH_ENQUIRY' in intent:
            tech_details = self.tech_prompt(user_input, tech_prompt)
            agent_response = f'Cung cấp thông tin chi tiết về kỹ thuật content: {tech_details}'
        elif 'METHOD_ENQUIRY' in intent:
            method_details = self.method_prompt(user_input, method_prompt)
            agent_response = f'Cung cấp thông tin chi tiết về phương pháp viết content: {method_details}'
        elif 'CONTEXT_ENQUIRY' in intent:
            context_details = self.context_prompt(user_input, context_prompt)
            agent_response = f'Viết bài content theo yêu cầu: {context_details}'
        else:
            agent_response = "Xin lỗi, tôi không thể hiểu được yêu cầu của bạn."

        self.history.append({'role': 'assistant', 'content': agent_response})
        self.save_history()

        # self.save_to_vectorstore(user_input, agent_response)

        return agent_response


