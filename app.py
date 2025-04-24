import streamlit as st
import os
import time
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import re
st.title("Chat with DeepSeek R1 on Ollama Demo")



# Initialize Ollama with DeepSeek R1 model
llm = Ollama(model="deepseek-r1:1.5b")

# Prompt for LLM
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Embedding function with feedback and error handling
def vector_embedding():
    if "vectors" not in st.session_state:
        # Check if directory exists and has files
        if not os.path.exists("./docus") or not os.listdir("./docus"):
            st.error("No documents found in './docus'. Please add some PDF files.")
            return

        st.write("Initializing embeddings...")
        st.session_state.embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")

        st.write("Loading documents...")
        st.session_state.loader = PyPDFDirectoryLoader("./docus")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents.")

        st.write("Splitting documents...")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.write(f"Created {len(st.session_state.final_documents)} chunks.")

        st.write("Creating FAISS vector store...")
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        st.write("✅ Vector store created successfully!")
    else:
        st.info("Vector store is already initialized.")

# Button to create embeddings
if st.button("Documents Embedding"):
    with st.spinner("Processing documents and creating vector store..."):
        try:
            vector_embedding()
        except Exception as e:
            st.error(f"Error during embedding: {e}")

# Input for asking questions
prompt1 = st.text_input("Enter Your Question About Documents")

# Query handling
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Documents Embedding' button first to create vector store.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        try:
            response = retrieval_chain.invoke({'input': prompt1})
            elapsed_time = time.process_time() - start

            st.write(f"🕒 Response time: {elapsed_time:.2f} seconds")
            st.write(f"💬 **Answer:** {response['answer']}")

            with st.expander("📄 Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

        except Exception as e:
            st.error(f"Error during response generation: {e}")
