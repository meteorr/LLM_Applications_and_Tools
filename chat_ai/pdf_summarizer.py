#!/usr/bin/env python
# coding: utf-8

# Install required libraries
# !pip install langchain
# !pip install streamlit
# !pip install PyPDF2
# !pip install langchain-openai

import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

api_key = "sk-"

# Function to process text into chunks and embeddings
def process_text(text): 
    # Split the text into chunks using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings using OpenAI's model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

# Main function to create a web app using Streamlit
def main():  
    st.title("📄 PDF Summarizer")
    st.divider()

    pdf = st.file_uploader('Upload a PDF file', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""   # Save the content of the PDF into the text variable
        for page in pdf_reader.pages:
            text += page.extract_text()

        documents = process_text(text)
        query = "Summarize the content of the uploaded PDF file into 3-5 sentences."  # Request summary of the PDF file

        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('--Summary Result--:')
            st.write(response)

if __name__ == '__main__':
    main()
