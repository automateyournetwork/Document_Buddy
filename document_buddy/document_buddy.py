import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredPowerPointLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker

# Load environment variables
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

# Message classes
class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass

class ChatWithFile:  # Renamed from ChatWithCSV
    def __init__(self, file_path, file_type):
        self.file_path = file_path
        self.file_type = file_type  # Accept file type as a parameter
        self.conversation_history = []
        self.load_file()
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_file(self):
        # Use the appropriate loader based on the file type
        if self.file_type == 'csv':
            self.loader = CSVLoader(file_path=self.file_path)
        elif self.file_type == 'pdf':
            self.loader = PyMuPDFLoader(file_path=self.file_path)
        elif self.file_type == 'txt':
            self.loader = TextLoader(file_path=self.file_path)
        elif self.file_type == 'pptx':
            self.loader = UnstructuredPowerPointLoader(file_path=self.file_path)
        elif self.file_type == 'docx':
            self.loader = Docx2txtLoader(file_path=self.file_path)
        elif self.file_type == 'xlsx':
            self.loader = UnstructuredExcelLoader(file_path=self.file_path, mode="elements")                                    
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        self.text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        # Convert complex metadata to string
        def simplify_metadata(doc):
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                for key, value in doc.metadata.items():
                    if isinstance(value, (list, dict)):
                        doc.metadata[key] = str(value)
            return doc

        # Simplify metadata for all documents
        self.docs = [simplify_metadata(doc) for doc in self.docs]

        # Proceed with storing documents in Chroma
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.qa_chains = []  # Initialize an empty list to hold the active QA chains

        # Check if the OpenAI API key is provided and set up the OpenAI conversational retrieval chain
        if openai_api_key:
            self.llm_openai = ChatOpenAI(api_key=openai_api_key, temperature=0.7, model="gpt-4-1106-preview")
            openai_qa = ConversationalRetrievalChain.from_llm(self.llm_openai, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)
            self.qa_chains.append(openai_qa)

        # Check if the Anthropic API key is provided and set up the Anthropic conversational retrieval chain
        if anthropic_api_key:
            self.llm_anthropic = ChatAnthropic(api_key=anthropic_api_key, temperature=0.7, model_name="claude-3-opus-20240229")
            anthropic_qa = ConversationalRetrievalChain.from_llm(self.llm_anthropic, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)
            self.qa_chains.append(anthropic_qa)

        # If neither API key is provided, you might want to raise an error or handle this case appropriately
        if not self.qa_chains:
            raise ValueError("No API keys provided for OpenAI or Anthropic. Please provide at least one.")

    def chat(self, question):
        responses = []

        # Check if OpenAI's API key is available and query OpenAI's GPT model
        if openai_api_key:
            response_openai = self.qa.invoke(question)
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=f"OpenAI's response: {response_openai.get('answer', 'Response not structured as expected.')}"))
            responses.append(response_openai)

        # Check if Anthropic's API key is available and query Anthropic's model
        if anthropic_api_key:
            response_anthropic = self.anthropic_qa.invoke(question)
            # No need to append the question again if OpenAI's response is already appended
            if not openai_api_key:
                self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=f"Anthropic's response: {response_anthropic.get('answer', 'Response not structured as expected.')}"))
            responses.append(response_anthropic)

        return responses

def upload_and_handle_file():
    st.title('Document Buddy - Chat with Document Data')
    uploaded_file = st.file_uploader("Choose a XLSX, PPTX, DOCX, PDF, CSV, or TXT file", type=["xlsx", "pptx", "docx", "pdf", "csv", "txt"])
    if uploaded_file:
        # Determine the file type and set accordingly
        if uploaded_file.name.endswith('.csv'):
            file_type = "csv"
        elif uploaded_file.name.endswith('.pdf'):
            file_type = "pdf"
        elif uploaded_file.name.endswith('.txt'):
            file_type = "txt"
        elif uploaded_file.name.endswith('.pptx'):
            file_type = "pptx"
        elif uploaded_file.name.endswith('.docx'):
            file_type = "docx"
        elif uploaded_file.name.endswith('.xlsx'):
            file_type = "xlsx"
        else:
            file_type = None  # Fallback in case of unexpected file extension

        if file_type:
            csv_pdf_txt_path = os.path.join("temp", uploaded_file.name)
            if not os.path.exists('temp'):
                os.makedirs('temp')
            with open(csv_pdf_txt_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state['file_path'] = csv_pdf_txt_path
            st.session_state['file_type'] = file_type  # Store the file type in session state
            st.success(f"{file_type.upper()} file uploaded successfully.")
            st.button("Proceed to Chat", on_click=lambda: st.session_state.update({"page": 2}))
        else:
            st.error("Unsupported file type. Please upload a XLSX, PPTX, DOCX, PDF, CSV, or TXT file.")

def chat_interface():
    st.title('Document Buddy - Chat with Document Data')
    file_path = st.session_state.get('file_path')
    file_type = st.session_state.get('file_type')
    if not file_path or not os.path.exists(file_path):
        st.error("File missing. Please go back and upload a file.")
        return

    if 'chat_instance' not in st.session_state:
        st.session_state['chat_instance'] = ChatWithFile(file_path=file_path, file_type=file_type)

    user_input = st.text_input("Ask a question about the document data:")
    if user_input and st.button("Send"):
        with st.spinner('Thinking...'):
            response_openai, response_anthropic = st.session_state['chat_instance'].chat(user_input)
            
            st.markdown("**Answers:**")
            
            # Display OpenAI's response
            if 'answer' in response_openai:
                st.markdown(f"**OpenAI's response:** {response_openai['answer']}")
            else:
                st.markdown("**OpenAI's response:** No specific answer found.")
                
            # Display Anthropic's response
            if 'answer' in response_anthropic:
                st.markdown(f"**Anthropic's response:** {response_anthropic['answer']}")
            else:
                st.markdown("**Anthropic's response:** No specific answer found.")

            # Display chat history
            st.markdown("**Chat History:**")
            for message in st.session_state['chat_instance'].conversation_history:
                prefix = "*You:* " if isinstance(message, HumanMessage) else "*AI:* "
                st.markdown(f"{prefix}{message.content}")

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    if st.session_state['page'] == 1:
        upload_and_handle_file()
    elif st.session_state['page'] == 2:
        chat_interface()
