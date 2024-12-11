#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import os

# Set OpenAI API key (retrieve from environment variable)
os.environ["OPENAI_API_KEY"] = "sk-"  # Replace with your OpenAI API key securely

# Streamlit app title
st.title("CSV File Analysis")

# File upload
st.write("Upload a CSV file:")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    if st.button("Upload"):
        with st.spinner("Processing the file..."):
            try:
                # Load the uploaded CSV file into a DataFrame
                df = pd.read_csv(uploaded_file)
                st.write("Dataframe Preview:")
                st.dataframe(df.head())

                # Initialize LangChain agent
                st.write("Initializing LangChain agent...")
                agent = create_pandas_dataframe_agent(
                    ChatOpenAI(temperature=0, model='gpt-4o'),
                    df,
                    verbose=False,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True,
                )
                st.session_state.conversation = agent
                st.success("Agent initialization complete")
            except Exception as e:
                st.error(f"Error occurred while processing the file: {e}")

# Handle user query
if user_query := st.chat_input("Enter your question:"):
    with st.chat_message("user"):
        st.write(user_query)  # Display the user's question

    if st.session_state.conversation:
        with st.spinner("Processing your question..."):
            try:
                # Use the LangChain agent to generate a response
                result = st.session_state.conversation.invoke(user_query)
                response = result.get("output", result)  # Extract the output if it's a dictionary

                # Display the response
                with st.chat_message("assistant"):
                    st.write(response)

                # Update chat history
                st.session_state.chat_history.append({"question": user_query, "answer": response})
            except Exception as e:
                st.error(f"Error occurred while processing your question: {e}")
    else:
        with st.chat_message("assistant"):
            st.write("Please upload a CSV file first.")
