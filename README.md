# 💬 Updated PDF Chatbot
A Streamlit application that allows users to upload PDF documents and engage in a conversational AI chat through a chat interface. Built using Streamlit, LangChain, ChromaDB, and local LLM models like Llama2 with Ollama.


# 🚀 Demo

1. **Upload PDFs**: Upload one or multiple PDF files.
2. **Ask Questions**: Enter your question about the PDF content.
3. **Get Responses**: The chatbot will provide relevant answers based on the document content.
4. **Demo Link**: https://github.com/pruthi-saksham/Updated-PDF-Chatbot/blob/main/app3.mp4
  - Visit the link
  - Click on `View raw`
  - The video will be downloaded to your local system.


# Features

- **PDF Uploading:** 
  - Supports multiple PDF uploads at once.
  - Each PDF is processed to extract its content for interaction.

- **Managing Sessions:**
    - *Session ID*: Each conversation is associated with a Session ID (default: `default_session`).
    - *Start a New Chat*: To reset the chat history or begin a new conversation, enter a new Session ID in the input field. Each session maintains its own chat history, allowing you to manage multiple conversations simultaneously.

- **Vector Store:** 
  - Uses ChromaDB to store vector embeddings of PDF text.
  - Efficient retrieval of relevant document sections for accurate responses.

- **Conversational History:**
  - Maintains chat history to ensure context-aware responses.
  - Past queries and answers are stored and used for reformulation of current queries.

- **LLM Integration:** 
  - Utilizes a local `llama2` model via Ollama for text generation.
  - Avoids the need for external API keys, offering a fully local solution.

- **Interactive Chat Interface:** 
  - Built using Streamlit for real-time interaction.
  - Users can ask questions directly about the uploaded PDFs and receive answers immediately.


## Run Locally

Ensure the following dependencies are installed and configured:

- **Python Version:** 3.12
- **Streamlit:** Latest version
- **Ollama Model:** Download and set up `llama2` model locally by running this command in yoy command prompt `ollama run llama2`.


- **Clone the project repository to your local machine**:
```bash
git clone https://github.com/pruthi-saksham/Updated-PDF-Chatbot
```

- **Go to the project directory**:
```bash
cd Updated-PDF-Chatbot
```

- **Install dependencies**:

```bash
pip install -r requirements.txt
```

- **Run the Main Streamlit App `app3.py`**:  
The project includes three code files: `app.py`, `app2.py`, and `app3.py`. The first two were initial versions, but due to some issues, the main file is `app3.py`, which offers better functionality and an improved interface, But feel free to explore the other two files. Run the following command to start the app:

```bash
streamlit run app3.py
```


# Tech Stack

1. **Streamlit**:
- Provides the interactive web interface for uploading PDFs and interacting with the chatbot.
- Easily deployable and well-suited for building quick prototypes and web apps.

2. **LangChain**:
- An open-source framework that simplifies the integration of language models with other components such as document retrievers and conversational agents.
- LangChain’s retrieval system is used to combine chat history with document retrieval, ensuring accurate, context-aware responses.
- Handles the process of chaining together models, retrieval systems, and prompts into a unified workflow for query handling.

3. **ChromaDB**:
- A vector database that allows efficient storage and retrieval of vectorized document embeddings.
- It helps index document chunks and enables the system to quickly retrieve relevant information based on user queries.

4. **HuggingFace Embeddings**:
- Sentence-transformers/paraphrase-MiniLM-L6-v2 is used to generate embeddings from the text in the uploaded PDFs.
- These embeddings are then stored in ChromaDB for fast and relevant document retrieval during interactions.

5. **PyPDFLoader**:
- A Python tool for loading PDF files and extracting their textual content.
- It allows text extraction from each page of the uploaded PDFs, making it possible to parse large documents for efficient searching and interaction.

6. **ChatOllama (Local LLM)**:
- The ChatOllama library integrates with the Llama2 model, a large language model (LLM) that runs locally.
- Ensures all processing stays within the local environment, avoiding the need for external API calls.
- It is designed for efficient and scalable local AI model hosting, ensuring that the chatbot can generate meaningful responses without relying on cloud services.

7. **LangChain RAG (Retrieval-Augmented Generation)**:
- Combines the power of document retrieval and language models to enhance response quality.
- The integration of retrieval chains and history-aware retrievers makes the model contextually aware of past interactions, improving accuracy and continuity in conversation.


# 🚀 About Me
*Hi, I’m Saksham Pruthi, an AI Engineer passionate about creating innovative AI-powered solutions. I specialize in Generative AI, designing systems that bridge cutting-edge research and practical applications. With expertise in various AI frameworks and an eye for scalable technology, I enjoy tackling challenging projects that drive real-world impact*.

# 🛠 Skills
- **Programming Languages**: Python, C++
- **Generative AI Technologies**: Proficient in deploying and fine-tuning a variety of LLMs including Llama2, GPT (OpenAI), Mistral, Gemini Pro using frameworks like Hugging Face, OpenAI and Groq. Expertise in NLP tasks like tokenization, sentiment analysis, summarization, and machine translation. Skilled in computer vision (CV) with models for image classification, object detection, and segmentation (YOLO). Expertise in MLOps, building and maintaining pipelines for model training and monitoring. Proficient in conversational AI with platforms LangChain. Skilled in synthetic data generation and code generation
- **Vector Databases and Embedding Libraries**: Proficient in ChromaDB and FAISS for efficient vector storage, retrieval, and similarity search.
- **Frameworks, Tools & Libraries**: LangChain, HuggingFace , OpenAI API, Groq, TensorFlow, PyTorch, Streamlit.
- **Databases**: MongoDB, ChromaDB
- **Version Control**: Proficient in using Git for version control and GitHub for collaborative development, repository management, and continuous integration workflows.
