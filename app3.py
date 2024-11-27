import streamlit as st
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
from langchain.schema import HumanMessage, AIMessage  # Import message types
import logging

# Logging setup
logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="PDF Conversational Chat", page_icon="💬")
st.title("Updated PDF Chatbot 💬")
st.caption("Upload PDFs and chat with their content")

# Load the LLM (cached for efficiency)
@st.cache_resource()
def load_llm():
    return ChatOllama(model="llama2")

llm = load_llm()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Initialize or get chat history
if "store" not in st.session_state:
    st.session_state.store = {}

# PDF processing
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with open(f"./temp_{uploaded_file.name}", "wb") as file:
            file.write(uploaded_file.getvalue())

        # Load PDF content
        try:
            loader = PyPDFLoader(f"./temp_{uploaded_file.name}")
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.error(f"Error processing '{uploaded_file.name}': {e}")
    
    if documents:
        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        # Setup retrieval chain
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Reformulate the question based on chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant answering questions based on context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Chat-like interface
        session_id = st.text_input("Session ID", value="default_session")
        chat_history = get_session_history(session_id).messages
        
        # Display existing chat
        for msg in chat_history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"  # Handle LangChain message types
            with st.chat_message(role):
                st.markdown(msg.content)

        # Input for new message
        if prompt := st.chat_input("Ask a question about your PDF..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            chat_history.append(HumanMessage(content=prompt))  # Store as HumanMessage
            
            # Generate response
            with st.spinner("Thinking..."):
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": session_id}}
                )
                answer = response["answer"]

            with st.chat_message("assistant"):
                st.markdown(answer)
            chat_history.append(AIMessage(content=answer))  # Store as AIMessage
else:
    st.warning("Please upload at least one PDF to start the conversation.")
