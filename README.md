# ChatPDF AI

ChatPDF AI is a Streamlit app for asking questions about uploaded PDF files. It uses LangChain to retrieve relevant document content and Groq to generate concise answers, with chat history available during the current app session.

## Requirements

- Python installed on your computer
- A Groq API key

## Setup on Windows

Open PowerShell in the project directory and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents virtual environment activation, you can run the app using the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the app

With the virtual environment activated:

```powershell
streamlit run app.py
```

Or, without activating it:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit opens the app in your browser. Enter your Groq API key in the password field, upload one or more PDF files, and submit a question about their contents.

## How it works

The app loads the uploaded PDFs, splits their text into chunks, creates embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and uses a Chroma vector store to retrieve context for answering questions. Groq chat completions are provided through LangChain's `ChatGroq` integration using the `openai/gpt-oss-20b` model.

Conversation history is kept in Streamlit session state for the current session; it is not configured for persistent storage. The Groq API key is entered in the app and should not be committed to source control.
