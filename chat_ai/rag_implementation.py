import os
from langchain.document_loaders import TextLoader
from pathlib import Path

# Set the file path based on the current file's directory
current_dir = Path(__file__).resolve().parent
file_path = current_dir.parent.parent / 'Data' / 'AI.txt'
documents = TextLoader(str(file_path)).load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split the document into chunks
def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

from langchain_openai import OpenAIEmbeddings
api_key = "sk-"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)

from langchain.vectorstores import Chroma
persist_dir = current_dir.parent.parent / 'Data'
db = Chroma.from_documents(docs, embeddings, persist_directory=str(persist_dir))

from langchain.chat_models import ChatOpenAI
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name, api_key=api_key)

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)

# Generate an answer to the query using RAG
query = "What is AI?"
matching_docs = db.similarity_search(query)
answer = chain.run(input_documents=matching_docs, question=query)
answer
