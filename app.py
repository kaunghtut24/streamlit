import dotenv
import streamlit as st
import fitz  # PyMuPDF

# LangChain imports (grouped for clarity)
from langchain import hub
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.text_splitters import RecursiveCharacterTextSplitter

# Additional imports (if needed)
import sys 

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("pysqlite3-binary is not installed. Chroma might not work correctly.")

# Assuming you have a .env file with necessary environment variables
# (e.g., OPENAI_API_KEY)
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Define a class to encapsulate document sections
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        if metadata is None:
            self.metadata = {}  # Default metadata as an empty dictionary
        else:
            self.metadata = metadata

# Function to process PDF and create vectorstore (cached)
@st.cache_data
def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read())
    text = ""
    for page in doc:
        text += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    documents = [Document(split) for split in splits]
    vectorstore = Chroma.from_documents(documents=documents, embedding=OpenAIEmbeddings())
    return documents, vectorstore

# Streamlit app interface
st.set_page_config(page_title="Chat with PDF", page_icon="", layout='centered')
st.markdown("<h3 style='background:#0284fe;padding:20px;border-radius:10px;text-align:center;'>Chat with PDF document</h3>",
            unsafe_allow_html=True)
st.markdown("")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Show progress bar while processing
    with st.spinner("Processing PDF..."):
        documents, vectorstore = process_pdf(uploaded_file)

    # Retrieval and generation
    retriever = vectorstore.as_retriever()

    prompt_template = """Use the following pieces of context to answer the user's question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}

    Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",  # You can experiment with different models here
    temperature=0.4,  # Adjust temperature for creativity vs. factual responses
    max_tokens=1024,  # Maximum number of tokens in the generated response
    n=1,  # Number of responses to generate (useful for sampling multiple answers)
)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="question",
        output_key="answer",
        prompt=prompt,
        document_prompt=PromptTemplate(template="{page_content}", input_variables=["page_content"]),
        chain_type_kwargs={"return_source_documents": True}
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("You can ask me questions about the document."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = qa_chain({"question": prompt})
            st.markdown(response['answer'])
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

    if st.button("Clear chat"):
        st.session_state.clear()
