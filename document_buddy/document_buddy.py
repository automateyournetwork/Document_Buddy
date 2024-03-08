import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader, PyMuPDFLoader, TextLoader, UnstructuredPowerPointLoader, Docx2txtLoader, UnstructuredExcelLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain.load import dumps, loads

# Load environment variables
load_dotenv()

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

class ChatWithFile:
    def __init__(self, file_path, file_type):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.file_path = file_path
        self.file_type = file_type
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
        self.llm = None
        self.llm_anthropic = None

        # Only initialize OpenAI's LLM if the API key is provided
        if self.openai_api_key:
            self.llm = ChatOpenAI(temperature=0.7, model="gpt-4-1106-preview", openai_api_key=self.openai_api_key)

        # Only initialize Anthropic's LLM if the API key is provided
        if self.anthropic_api_key:
            self.llm_anthropic = ChatAnthropic(temperature=0.7, model_name="claude-3-opus-20240229", anthropic_api_key=self.anthropic_api_key)

        if self.llm:
            self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)
        if self.llm_anthropic:
            self.anthropic_qa = ConversationalRetrievalChain.from_llm(self.llm_anthropic, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def chat(self, question):
        related_queries = self.generate_related_queries(question)
        queries = [question] + related_queries

        all_results = []

        for query in queries:
            response = None
            if self.llm:
                response = self.qa.invoke({"question": query})
            elif self.llm_anthropic:
                response = self.anthropic_qa.invoke({"question": query})

            if response:
                # Here, the response should already be leveraging your ChromaDB
                # and the document data it contains for generating answers.
                st.write("Query:", query)
                st.write("Response:", response.answer)
                all_results.append(response)
            else:
                st.write("No response received.")

        # Rank the results using RRF
        st.write("All results before RRF:", all_results)                    
        ranked_results = self.reciprocal_rank_fusion(all_results)
        st.write("Ranked Results:", ranked_results)
        return ranked_results

    def generate_related_queries(self, original_query):
        prompt = f"Given the original query: '{original_query}', generate a JSON array of four related search queries."
        response = self.llm.invoke(input=prompt)

        if hasattr(response, 'content'):
            # Directly access the 'content' if the response is the expected object
            generated_text = response.content
        elif isinstance(response, dict) and 'content' in response:
            # Extract 'content' if the response is a dict
            generated_text = response['content']
        else:
            # Fallback if the structure is different or unknown
            generated_text = str(response)
            st.error("Unexpected response format.")

        st.write("Response content:", generated_text)

        # Assuming the 'content' starts with "content='" and ends with "'"
        # Attempt to directly parse the JSON part, assuming no other wrapping
        try:
            json_start = generated_text.find('[')
            json_end = generated_text.rfind(']') + 1
            json_str = generated_text[json_start:json_end]
            related_queries = json.loads(json_str)
            st.write("Parsed related queries:", related_queries)
        except (ValueError, json.JSONDecodeError) as e:
            st.error(f"Failed to parse JSON: {e}")
            related_queries = []

        return related_queries

    def retrieve_documents(self, query):
        # Example: Convert query to embeddings and perform a vector search in ChromaDB
        query_embedding = OpenAIEmbeddings()  # Assuming SemanticChunker can embed text
        search_results = self.vectordb.search(query_embedding, top_k=5)  # Adjust based on your setup
        document_ids = [result['id'] for result in search_results]  # Extract document IDs from results
        return document_ids

    def reciprocal_rank_fusion(self, results_lists, k=60):
        fused_scores = {}
        for rank, results in enumerate(results_lists):
            if isinstance(results, dict):
                # Using the query as a unique identifier for each set of results
                doc_id = results.get("question")
                score = fused_scores.get(doc_id, {"score": 0})["score"]
                fused_scores[doc_id] = {"doc": results, "score": score + 1 / (rank + 1 + k)}

        # Sort by fused score
        reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

        return [entry["doc"] for entry in reranked_results]

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
            all_answers = st.session_state['chat_instance'].chat(user_input)
            
            # Display all collected answers
            st.markdown("**Answers:**")
            for answer in all_answers:
                st.markdown(answer)
                
            # Display chat history (You might want to adjust how or if you display this based on the new output)
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
