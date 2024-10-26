import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.llms import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from docx import Document
import fitz
import os

load_dotenv()

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_vectorstore_from_word(file):
    doc = Document(file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    document_text = '\n'.join(full_text)

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_text(document_text)
    
    vector_store = Chroma.from_texts(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_vectorstore_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        full_text.append(page.get_text("text"))
    document_text = '\n'.join(full_text)

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_text(document_text)
    
    vector_store = Chroma.from_texts(document_chunks, OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = OpenAI()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = OpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, vector_store, chat_history_key):
    retriever_chain = get_context_retriever_chain(vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state[chat_history_key],
        "input": user_input
    })
    
    return response['answer']

def main():
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set")
        return

    st.set_page_config(page_title="Chat with websites, PDFs, or Word Documents", page_icon="🤖")
    st.title("Chat with websites, PDFs, or Word Documents")

    option = st.sidebar.selectbox("Choose an option", ["Chat with Website", "Chat with PDF", "Chat with Word Document"])

    if option == "Chat with Website":
        st.header("Chat with Website")
        website_url = st.text_input("Website URL")

        if website_url:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
            else:
                st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

            st.session_state.vector_store = get_vectorstore_from_url(website_url)    

            user_query = st.chat_input("Type your message here...")
            if user_query:
                response = get_response(user_query, st.session_state.vector_store, "chat_history")
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                st.session_state.chat_history.append(AIMessage(content=response))

            for message in st.session_state.chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)

    elif option == "Chat with PDF":
        st.header("Chat with PDF 📄")

        pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
        if pdf_file:
            if "pdf_chat_history" not in st.session_state:
                st.session_state.pdf_chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
            else:
                st.session_state.pdf_chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

            st.session_state.vector_store = get_vectorstore_from_pdf(pdf_file)    

            user_query = st.chat_input("Type your message here...")
            if user_query:
                response = get_response(user_query, st.session_state.vector_store, "pdf_chat_history")
                st.session_state.pdf_chat_history.append(HumanMessage(content=user_query))
                st.session_state.pdf_chat_history.append(AIMessage(content=response))

            for message in st.session_state.pdf_chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)

    elif option == "Chat with Word Document":
        st.header("Chat with Word Document 📄")

        word_file = st.file_uploader("Upload a Word document", type="docx")
        if word_file:
            if "word_chat_history" not in st.session_state:
                st.session_state.word_chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
            else:
                st.session_state.word_chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

            st.session_state.vector_store = get_vectorstore_from_word(word_file)    

            user_query = st.chat_input("Type your message here...")
            if user_query:
                response = get_response(user_query, st.session_state.vector_store, "word_chat_history")
                st.session_state.word_chat_history.append(HumanMessage(content=user_query))
                st.session_state.word_chat_history.append(AIMessage(content=response))

            for message in st.session_state.word_chat_history:
                if isinstance(message, AIMessage):
                    with st.chat_message("AI"):
                        st.write(message.content)
                elif isinstance(message, HumanMessage):
                    with st.chat_message("Human"):
                        st.write(message.content)

if __name__ == "__main__":
    main()
