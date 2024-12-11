#!/usr/bin/env python
# coding: utf-8

# Install necessary packages
# !pip install langchain
# !pip install streamlit
# !pip install openai

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import os

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-"  # Enter your OpenAI API key here

# Web page content
langs = ["Korean", "Japanese", "Chinese", "English"]  # List of languages for translation
left_co, cent_co, last_co = st.columns(3)

# Sidebar for selecting language
with st.sidebar:
    language = st.radio('Select the language you want to translate into:', langs)

st.markdown('### Language Translation Service')
prompt = st.text_input('Enter the text you want to translate')  # User text input

# Template for translation prompt
trans_template = PromptTemplate(
    input_variables=['trans'],
    template='Your task is to translate this text to ' + language + '. TEXT: {trans}'
)

# Memory for storing text
memory = ConversationBufferMemory(input_key='trans', memory_key='chat_history')

# Initialize the language model and translation chain
llm = ChatOpenAI(temperature=0.0, model_name='gpt-4')
trans_chain = LLMChain(llm=llm, prompt=trans_template, verbose=True, output_key='translate', memory=memory)

# Process the input text and display the translated response
if st.button("Translate"):
    if prompt:
        response = trans_chain({'trans': prompt})
        st.info(response['translate'])
