import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import time
persist_directory = 'db'
vectordb = None
vectorstore = None

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,separators=['\n', ' ', '.', ',', ';', ':', '?', '!'])
    chunks = text_splitter.split_text(text)
    return chunks

def vectordb1(text_chunks):
    embedding = OpenAIEmbeddings()
    vectordb = None
    # vectordb.delete_collection()
    if vectordb is not None:
        vectordb.close()  # Close the existing database
        vectordb = None
        vectordb.delete_collection()
        vectordb.persist()
    vectordb = Chroma.from_texts(texts=text_chunks,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
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

def get_compl(prompt, model= "gpt-3.5-turbo"):
    
    op = openai.ChatCompletion.create(model=model,
        messages=[{"role": "system", "content": "you are a helpful assistant"},
                {"role": "user", "content": prompt}],
        temperature=0,)
    return str(op.choices[0].message["content"])

def speech_to_text(audio):
    if audio is not None:
        result = openai.Audio.transcribe("whisper-1", audio, verbose=True, api_key=os.getenv("OPENAI_API_KEY"))
        return result

def main():
    load_dotenv()
    st.set_page_config(page_title="Augmented RAG", page_icon=":hehe:", layout="wide", initial_sidebar_state="expanded",)
    audio_uploaded = False
    re = get_compl('hii')
    
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

    st.header("Retrival augmented chatbot - referencing the data uploaded:")

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
        if audio_uploaded:
            response = inference(prompt, st.session_state.vectorstore)['result']
        else:
            response = get_compl(prompt)
        # response = inference(prompt, st.session_state.vectorstore)['result']
        # Display assistant response in chat message container
        #add time delay
        time.sleep(2)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):

            st.markdown(f'd{response}')
        # Add assistant response to chat history
        




if __name__ == '__main__':
    main()
