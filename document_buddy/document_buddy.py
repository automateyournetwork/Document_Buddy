"""
Main application
"""
import logging
import os
import json
import pathlib
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
# from langchain.load import dumps, loads
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.environ.get("SECRETS_PATH"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

FILE_LOADERS = {
    "csv": CSVLoader,
    "docx": Docx2txtLoader,
    "pdf": PyMuPDFLoader,
    "pptx": UnstructuredPowerPointLoader,
    "txt": TextLoader,
    "xlsx": UnstructuredExcelLoader,
}

ACCEPTED_FILE_TYPES = list(FILE_LOADERS)


# Message classes
class Message:
    """
    Base message class
    """
    def __init__(self, content):
        self.content = content


class HumanMessage(Message):
    """
    Represents a message from the user.
    """


class AIMessage(Message):
    """
    Represents a message from the AI.
    """


class ChatWithFile:
    """
    Main class...?
    """
    def __init__(self, file_path, file_type):
        """
        Perform initial parsing of the uploaded file and initialize the
        chat instance.

        :param file_path: Full path and name of uploaded file
        :param file_type: File extension determined after upload
        """
        loader = FILE_LOADERS[file_type](file_path=file_path)
        pages = loader.load_and_split()
        docs = self.split_into_chunks(pages)
        self.store_in_chroma(docs)

        self.conversation_history = []

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4-1106-preview",
            openai_api_key=OPENAI_API_KEY
        )

        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.vectordb.as_retriever(search_kwargs={"k": 10}),
            memory=self.memory
        )

    @staticmethod
    def split_into_chunks(pages):
        """
        Split the document pages into chunks based on similarity

        :return: Result of langchain_experimental.text_splitter.SemanticChunker
        """
        text_splitter = SemanticChunker(
            embeddings=OpenAIEmbeddings(),
            breakpoint_threshold_type="percentile"
        )
        return text_splitter.split_documents(pages)

    @staticmethod
    def simplify_metadata(doc):
        """
        If the provided doc contains a metadata dict, iterate over the
        metadata and ensure values are stored as strings.

        :param doc: Chunked document to process
        :return: Document with any metadata values cast to string
        """
        metadata = getattr(doc, "metadata", None)
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    metadata[key] = str(value)
        return doc

    def store_in_chroma(self, docs):
        """
        Store each document in Chroma
        """
        # Convert complex metadata to string
        # def simplify_metadata(doc):
        #     metadata = getattr(doc, "metadata", None)
        #     if isinstance(metadata, dict):
        #         for key, value in metadata.items():
        #             if isinstance(value, (list, dict)):
        #                 metadata[key] = str(value)
        #     return doc

        # Simplify metadata for all documents
        docs = [self.simplify_metadata(doc) for doc in docs]

        # Proceed with storing documents in Chroma
        # embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
        self.vectordb.persist()

    def chat(self, question):
        # Generate related queries based on the initial question
        related_queries_dicts = self.generate_related_queries(question)
        # Ensure that queries are in string format, extracting the 'query' value from dictionaries
        related_queries = [q["query"] for q in related_queries_dicts]
        # Combine the original question with the related queries
        queries = [question] + related_queries

        all_results = []

        for idx, query_text in enumerate(queries):
            response = None
            response = self.qa.invoke(query_text)

            # Process the response
            if response:
                st.write("Query: ", query_text)
                st.write("Response: ", response["answer"])
                all_results.append(
                    {
                        "query": query_text,
                        "answer": response['answer']
                    }
                )
            else:
                st.write("No response received for: ", query_text)

        # After gathering all results, let's ask the LLM to synthesize a comprehensive answer
        if all_results:
            # Assuming reciprocal_rank_fusion is correctly applied and scored_results is prepared
            reranked_results = self.reciprocal_rank_fusion(all_results)
            # Prepare scored_results, ensuring it has the correct structure
            scored_results = [{"score": res["score"], **res["doc"]} for res in reranked_results]
            synthesis_prompt = self.create_synthesis_prompt(question, scored_results)
            synthesized_response = self.llm.invoke(synthesis_prompt)

            if synthesized_response:
                # Assuming synthesized_response is an AIMessage object with a 'content' attribute
                st.write(synthesized_response)
                final_answer = synthesized_response.content
            else:
                final_answer = "Unable to synthesize a response."

            # Update conversation history with the original question and the synthesized answer
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=final_answer))

            return {"answer": final_answer}

        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content="No answer available."))
        return {"answer": "No results were available to synthesize a response."}

    def generate_related_queries(self, original_query):
        # NOTE: This prompt is split on sentences for readability. No newlines
        # will be included in the output due to implied line continuation.
        prompt = (
            f"In light of the original inquiry: '{original_query}', let's "
            "delve deeper and broaden our exploration. Please construct a "
            "JSON array containing four distinct but interconnected search "
            "queries. Each query should reinterpret the original prompt's "
            "essence, introducing new dimensions or perspectives to "
            "investigate. Aim for a blend of complexity and specificity in "
            "your rephrasings, ensuring each query unveils different facets "
            "of the original question. This approach is intended to "
            "encapsulate a more comprehensive understanding and generate the "
            "most insightful answers possible. Only respond with the JSON "
            "array itself."
        )
        response = self.llm.invoke(input=prompt)

        if hasattr(response, "content"):
            # Directly access the 'content' if the response is the expected object
            generated_text = response.content
        elif isinstance(response, dict) and "content" in response:
            # Extract 'content' if the response is a dict
            generated_text = response["content"]
        else:
            # Fallback if the structure is different or unknown
            generated_text = str(response)
            st.error("Unexpected response format.")

        #st.write("Response content:", generated_text)

        # Assuming the 'content' starts with "content='" and ends with "'"
        # Attempt to directly parse the JSON part, assuming no other wrapping
        try:
            json_start = generated_text.find('[')
            json_end = generated_text.rfind(']') + 1
            json_str = generated_text[json_start:json_end]
            related_queries = json.loads(json_str)
            #st.write("Parsed related queries:", related_queries)
        except (ValueError, json.JSONDecodeError) as e:
            #st.error(f"Failed to parse JSON: {e}")
            related_queries = []

        return related_queries

    def retrieve_documents(self, query):
        # pylint: disable=line-too-long
        # Example: Convert query to embeddings and perform a vector search in ChromaDB
        query_embedding = OpenAIEmbeddings()  # Assuming SemanticChunker can embed text
        search_results = self.vectordb.search(query_embedding, top_k=5)  # Adjust based on your setup
        document_ids = [result["id"] for result in search_results]  # Extract document IDs from results
        return document_ids

    def reciprocal_rank_fusion(self, all_results, k=60):
        # Assuming each result in all_results can be uniquely identified for scoring
        # And assuming all_results is directly the list you want to work with
        fused_scores = {}
        for result in all_results:
            # Let's assume you have a way to uniquely identify each result; for simplicity, use its index
            doc_id = result["query"]  # or any unique identifier within each result
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": result, "score": 0}
            # Example scoring adjustment; this part needs to be aligned with your actual scoring logic
            fused_scores[doc_id]["score"] += 1  # Simplified; replace with actual scoring logic

        reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return reranked_results

    def create_synthesis_prompt(self, original_question, all_results):
        # Sort the results based on RRF score if not already sorted; highest scores first
        sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        st.write("Sorted Results", sorted_results)
        prompt = (
            f"Based on the user's original question: '{original_question}', "
             "here are the answers to the original and related questions, "
             "ordered by their relevance (with RRF scores). Please synthesize "
             "a comprehensive answer focusing on answering the original "
             "question using all the information provided below:\n\n"
        )

        # Include RRF scores in the prompt, and emphasize higher-ranked answers
        for idx, result in enumerate(sorted_results):
            prompt += f"Answer {idx+1} (Score: {result['score']}): {result['answer']}\n\n"

        prompt += (
            "Given the above answers, especially considering those with "
            "higher scores, please provide the best possible composite answer "
            "to the user's original question."
        )

        return prompt


def upload_and_handle_file():
    st.title("Document Buddy - Chat with Document Data")
    uploaded_file = st.file_uploader(
        label=(
            f"Choose a {', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
            f"{ACCEPTED_FILE_TYPES[-1].upper()} file"
        ),
        type=ACCEPTED_FILE_TYPES
    )
    if uploaded_file:
        # Determine the file type and set accordingly
        file_type = pathlib.Path(uploaded_file.name).suffix
        file_type = file_type.replace(".", "")

        if file_type:  # Will be an empty string if no extension
            csv_pdf_txt_path = os.path.join("temp", uploaded_file.name)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            with open(csv_pdf_txt_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state["file_path"] = csv_pdf_txt_path
            st.session_state["file_type"] = file_type  # Store the file type in session state
            st.success(f"{file_type.upper()} file uploaded successfully.")
            st.button(
                "Proceed to Chat",
                on_click=lambda: st.session_state.update({"page": 2})
            )
        else:
            st.error(
                f"Unsupported file type. Please upload a {', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
                f"{ACCEPTED_FILE_TYPES[-1].upper()} file."
            )


def chat_interface():
    st.title("Document Buddy - Chat with Document Data")
    file_path = st.session_state.get("file_path")
    file_type = st.session_state.get("file_type")
    if not file_path or not os.path.exists(file_path):
        st.error("File missing. Please go back and upload a file.")
        return

    if "chat_instance" not in st.session_state:
        st.session_state["chat_instance"] = ChatWithFile(
            file_path=file_path,
            file_type=file_type
        )

    user_input = st.text_input("Ask a question about the document data:")
    if user_input and st.button("Send"):
        with st.spinner("Thinking..."):
            top_result = st.session_state["chat_instance"].chat(user_input)

            # Display the top result's answer as markdown for better readability
            if top_result:
                st.markdown("**Top Answer:**")
                st.markdown(f"> {top_result['answer']}")
            else:
                st.write("No top result available.")

            # Display chat history
            st.markdown("**Chat History:**")
            for message in st.session_state["chat_instance"].conversation_history:
                prefix = "*You:* " if isinstance(message, HumanMessage) else "*AI:* "
                st.markdown(f"{prefix}{message.content}")


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        st.title("Document Buddy - Chat with Document Data")
        st.error(
            "No API keys have been defined - verify settings in the .env "
            "file and restart the app using Docker Compose."
        )
    else:
        if "page" not in st.session_state:
            st.session_state["page"] = 1

    if st.session_state["page"] == 1:
        upload_and_handle_file()
    elif st.session_state["page"] == 2:
        chat_interface()
