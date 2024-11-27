import streamlit as st
import os
import logging
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

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit UI setup
st.title("Conversational RAG with PDF Upload and Chat History")
st.write("Upload PDF and chat with its content")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the local LLM using Ollama
llm = ChatOllama(model="llama2")  # Adjust the model name according to your Ollama setup

# Session management
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

# File uploader for PDFs
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = f"./temp_{uploaded_file.name}"
        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getvalue())

        try:
            # Load PDF content
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {str(e)}")
            logger.error(f"Failed to load {uploaded_file.name}: {str(e)}")
        finally:
            os.remove(temp_path)  # Clean up temp file

    # Split and embed documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create Chroma vector store
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()

    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, reformulate it to be a standalone question."),
        MessagesPlaceholder(variable_name="chat_history"),  # Set explicitly
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # QA prompt template with context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an assistant for question-answering tasks. "
         "Use the following context to answer the question. If unsure, say you don't know. Keep it concise.\n\n"
         "Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Build RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Manage chat history
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

    # User input for questions
    user_input = st.text_input("Your question:")
    if user_input:
        with st.spinner("Thinking..."):
            session_history = get_session_history(session_id)
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.success("Assistant: " + response["answer"])
                st.write("Chat history:", [msg.content for msg in session_history.messages])
            except Exception as e:
                st.error(f"Error during inference: {str(e)}")
                logger.error(f"Inference error: {str(e)}")
else:
    st.warning("Please upload at least one PDF to start the conversation.")
