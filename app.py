import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import whisper
from PyPDF2 import PdfReader
import time
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

# model = whisper.load_model("base")

persist_directory = 'db'

vectorstore = None
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,
                                                   separators=['\n', ' ', '.', ',', ';', ':', '?', '!'])
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def vectordb1(text_chunks, query="do nothing"):
    embedding = OpenAIEmbeddings()
    vectordb = None
    
    vectordb = Chroma.from_texts(texts=text_chunks,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
    
    # vectordb.persist()
    # vectordb = None
    # vectordb = Chroma(persist_directory=persist_directory,
    #                   embedding_function=embedding)
    return vectordb


def inference(query, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,)
    querys = query
    llm_response = qa_chain(querys)
    return llm_response

def get_completion(prompt, model= "gpt-3.5-turbo"):
    messages = [{"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response

def speech_to_text(audio):
    if audio is not None:
        result = openai.Audio.transcribe("whisper-1", audio, verbose=True, api_key=os.getenv("OPENAI_API_KEY"))
        return result

def com(prompt):
    return get_completion(prompt, model= "gpt-3.5-turbo")


def main():
    load_dotenv()
    st.set_page_config(page_title="Augmented RAG", page_icon=":hehe:", layout="wide", initial_sidebar_state="expanded",)
    
    # Create a flag to check if an audio file has been uploaded
    audio_uploaded = False
    
    with st.sidebar:
        st.header("Upload your Audio or Documents here:")
        audio = st.file_uploader("Upload an audio file", type=["mp3"])
        pdf_doc = st.file_uploader("Upload your document here:", accept_multiple_files=True)

        # If an audio file is uploaded, set the flag to True
        if audio is not None:
            audio_uploaded = True

        # Use the flag to conditionally activate the chat
        if audio_uploaded:
            if st.button("Submit"):
                with st.spinner("Processing..."):
                    raw_text1 = str(speech_to_text(audio)['text'])
                    raw_text = ""
                    if pdf_doc is not None:
                        for pdf in pdf_doc:
                            pdf_text = PdfReader(pdf)
                            pdf_text_content = ""
                            for page in pdf_text.pages:
                                pdf_text_content += page.extract_text()
                            raw_text += pdf_text_content
                    raw_text2 = raw_text1 + raw_text
                    
                    chunks1 = get_text_chunks(raw_text2)
                    st.session_state.vectorstore = vectordb1(chunks1)
        else:
            st.write("Please upload an audio file to activate the chat.")
    
    # Conditionally render the chat only if an audio file has been uploaded
    if audio_uploaded:
        st.header("Retrival augmented chatbot - referencing the data uploaded:")

        # Initialize chat history
        if "messages" not in st.session_state1:
            st.session_state.messages1 = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages1:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = inference(prompt, st.session_state.vectorstore)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response['result'])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response['result']})

    else:
        st.header("normal")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = com(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(str(response.choices[0].message["content"]))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message["content"]})
        # st.header("Normal chatbot")
        # if "messages" not in st.session_state:
        #     st.session_state.messages = []
        # print('Session state', st.session_state.messages)
        # # Display chat messages from history on app rerun
        # for message in st.session_state.messages:
        #     with st.chat_message(message["role"]):
        #         st.markdown(message["content"])

        # # React to user input
        # if prompt := st.chat_input("What is up?"):
        #     print(prompt)
        #     # Display user message in chat message container
        #     st.chat_message("user").markdown(prompt)
        #     # Add user message to chat history
        #     st.session_state.messages.append({"role": "user", "content": prompt})
        #     # print('hh')
        #     # prin
        #     # response = 'hllo'
        #     response1 = com(prompt)
        #     print(response1)
        #     st.session_state.messages.append({"role": "assistant", "content": response1})


        #     with st.chat_message("assistant"):
        #         print(st.session_state.messages)
        #         st.markdown(response1)
            # Add assistant response to chat history




if __name__ == '__main__':
    main()
