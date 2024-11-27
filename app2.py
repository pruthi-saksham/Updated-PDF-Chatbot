import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
import PyPDF2
import logging
import os

# Logging setup
logging.basicConfig(level=logging.WARNING)

st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload PDFs and chat with their content")

@st.cache_resource()
def load_llm():
    return ChatOllama(model="llama2")

llm = load_llm()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

session_id = st.text_input("Session ID", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

# PDF validation function
def is_valid_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            PyPDF2.PdfReader(f)
        return True
    except PyPDF2.errors.PdfReadError:
        return False

# Process uploaded files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.size == 0:
            st.warning(f"The file '{uploaded_file.name}' is empty. Please upload a valid PDF.")
            continue

        temppdf = "./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())

        # Validate PDF integrity
        if not is_valid_pdf(temppdf):
            st.error(f"The file '{uploaded_file.name}' is not a valid PDF.")
            continue
        
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    
    if not documents:
        st.error("No valid PDFs found. Please upload valid files.")
    else:
        # Split and embed documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
        vectorstore.persist()
        retriever = vectorstore.as_retriever()

        # Set up the QA chain
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, reformulate it to be a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the retrieved context to answer concisely.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User interaction
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.success("Assistant: " + response["answer"])
            st.write("Chat history:", session_history.messages)
else:
    st.warning("Please upload a PDF to start the conversation.")
