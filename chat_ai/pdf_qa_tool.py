#!/usr/bin/env python
# coding: utf-8

# Install required libraries
# !pip install langchain
# !pip install streamlit
# !pip install PyPDF2
# !pip install langchain-openai

import streamlit as st 
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

api_key = "sk-"

# Extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split the given text into smaller chunks based on specified conditions
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Generate embeddings for the given text chunks and create a vector store using FAISS
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Initialize a conversational chain with the given vector store
def get_conversation_chain(vectorstore):
    memory = ConversationBufferWindowMemory(memory_key='chat_history', return_message=True)  # Save previous conversations in memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name='gpt-4o', api_key=api_key),
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h: h,
        memory=memory
    )  # Send queries to LangChain chatbot using ConversationalRetrievalChain
    return conversation_chain

# Streamlit app for PDF upload and Q&A
user_uploads = st.file_uploader("Upload your files", accept_multiple_files=True)
if user_uploads is not None:
    if st.button("Upload"):
        with st.spinner("Processing..."):
            # Extract text from PDFs
            raw_text = get_pdf_text(user_uploads)
            # Split text into chunks
            text_chunks = get_text_chunks(raw_text)
            # Create a FAISS vector store to save PDF text
            vectorstore = get_vectorstore(text_chunks)
            # Create a conversational chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

if user_query := st.chat_input("Enter your question here"):
    # Process user's message using the conversation chain
    if 'conversation' in st.session_state:
        result = st.session_state.conversation({
            "question": user_query, 
            "chat_history": st.session_state.get('chat_history', [])
        })
        response = result["answer"]
    else:
        response = "Please upload a document first."
    with st.chat_message("assistant"):
        st.write(response)
