import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Đặt API key của OpenAI
os.environ["OPENAI_API_KEY"] = ''
embeddings = OpenAIEmbeddings()

def text_load(files):
    documents = []
    for file in files:
        loader = TextLoader(file, encoding='utf-8')
        docs = loader.load()
        documents.extend(docs)
    return documents

def get_chunk(documents):
    text_split = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=48)
    split_txt = text_split.split_documents(documents)
    return split_txt

def vector_data(text_chunks):
    vector_store = Qdrant.from_documents(
        text_chunks,
        embedding=embeddings,
        path="qdrant_database",
        collection_name='documents',
    )
    return vector_store

def load_vector(text_chunks, collection_name):
    client = QdrantClient(path="./qdrant_database")
    db = Qdrant(collection_name=collection_name,
                embeddings=embeddings, client=client, text_chunks = text_chunks)
    return db

def retrieve_knowledge(query, collection_name, file_txt):
    # Generate embeddings for the query
    query_vector = embeddings.embed_query(query)

    # Connect to Qdrant and search for the top 3 closest vectors
    client = QdrantClient(path="./qdrant_database")
    file_txt = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=3
    )

    return file_txt