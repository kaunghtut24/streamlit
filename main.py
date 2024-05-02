import dotenv
import streamlit as st
import fitz  # PyMuPDF
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    print("pysqlite3-binary is not installed. Chroma might not work correctly.")

dotenv.load_dotenv()

# Define a class to encapsulate document sections
class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        if metadata is None:
            self.metadata = {}  # Default metadata as an empty dictionary
        else:
            self.metadata = metadata

# Streamlit app interface
st.set_page_config(page_title="Chat with PDF",page_icon="",layout='centered')
st.markdown("<h3 style='background:#0284fe;padding:20px;border-radius:10px;text-align:center;'>Chat with PDF document</h3>",
        unsafe_allow_html=True)
st.markdown("")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    doc = fitz.open(stream=uploaded_file.read())

    # Initialize vectorstore before processing pages
    vectorstore = Chroma.from_documents(documents=[], embedding=OpenAIEmbeddings())

    # Process each page individually
    for page in doc:
        page_text = page.get_text()
        page_splits = text_splitter.split_text(page_text)

        # Create Document objects with page metadata
        page_documents = [Document(split, {"page": page.number}) for split in page_splits]

        # Add documents to the vectorstore directly
        vectorstore.add_documents(page_documents)

    # Retrieval and generation
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # React to user input
    if prompt := st.chat_input("You can ask me questions about document."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("Clear chat"):
        st.session_state.clear()


