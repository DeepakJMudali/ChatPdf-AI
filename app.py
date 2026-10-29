import streamlit as st
import os
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
import uuid
from dotenv import load_dotenv
load_dotenv()



## setup streamlit  

st.title("ChatPDF AI")
st.write("upload pdf's and chat with their content")

## Input the GROQ API KEY

groq_api_key = st.text_input("Enter your GROQ API key:",type="password")

## check if groq api key is provided


@st.cache_resource
def create_llm(api_key):
   
        return ChatGroq(model="openai/gpt-oss-20b",groq_api_key=api_key)

if "store" not in st.session_state:
    st.session_state["store"] ={}

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

uploaded_files = st.file_uploader("Choose a PDF file",type="pdf", accept_multiple_files=True)

## process my uploaded files

def create_documents():

     documents=[]

     for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
               file.write(uploaded_file.getvalue())
               file_name=uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)
     return documents


def create_vectorDB():
            docs= create_documents()
            splitter = RecursiveCharacterTextSplitter()
            chunk = splitter.split_documents(docs)
            embedder  =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorDB = Chroma.from_documents(chunk,embedder)
            return vectorDB

## This prompt is used to reformulate the query and get search from vectordb by retriever.
contextualize_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return  as it is."
        )

context_prompt = ChatPromptTemplate.from_messages([
      ("system",contextualize_system_prompt),
      MessagesPlaceholder("chat_history"), ## for retrieval
      ("human","{input}")
      
      ])

## It  will use context to answer and uses conversation history as well.

qa_prompt = ChatPromptTemplate.from_messages([("system","""
You are an assistant for question-answering task.
Use the following retrieved context to answer the question.
If you don't know the answer say that you don't know.
Use maximum three sentences and keep the answer concise. 
{context}
"""),
MessagesPlaceholder("chat_history"), ## for conversation
("human","{input}")]
)


def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
## create_history_aware_retriever uses previous conversation history context to retrieve the data from vectorDB

if groq_api_key and uploaded_files:
    llm = create_llm(groq_api_key)
    retriever = create_vectorDB().as_retriever()
    ## llm — it will reformulate the user's query with chat history if needed based on provided instruction in prompt.
    ## retriever- It will retrive the relevant chunks by using reformulated query.
    ## prompt guides to llm to use chat history if needed.

    ## 1. create_history_aware_retriever(): The LLM reformulates the user's query using chat history, if necessary.
    ## 2. Retriever: Searches the vector database using the reformulated query.
    ## 3. If relevant documents are found: They are passed to the LLM to generate the final answer.
    ## 4. If the PDF doesn't contain the answer: The LLM will perform action based on provided instruction to llm model using prompt.
    
    history_aware_retriever = create_history_aware_retriever(llm,retriever,context_prompt) ## it will retrive the documents using reformulated query when conversation history is available.
    document_chain = create_stuff_documents_chain(llm,qa_prompt)
    retriever_chain = create_retrieval_chain(history_aware_retriever,document_chain)
    chat_with_history = RunnableWithMessageHistory(retriever_chain,
                                               get_session_history,
                                               input_messages_key="input",
                                               output_messages_key="answer",
                                               history_messages_key="chat_history")

    user_input =st.text_input("your question:")

    if user_input:
      session_id = st.session_state["session_id"]
      session_history = get_session_history(session_id)
      config = {"configurable":{"session_id":session_id}}
      response = chat_with_history.invoke({"input": user_input},config)
      st.success(response["answer"])

else:
      st.warning("Please enter your GROQ API KEY")


