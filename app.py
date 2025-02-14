import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from chromadb.config import Settings
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import datetime
import logging  # Import logging
from langchain_core.messages import HumanMessage, AIMessage # ADD THIS LINE: Import HumanMessage and AIMessage


# --- Configuration ---
PERSIST_ROOT = "./chroma_databases"
os.makedirs(PERSIST_ROOT, exist_ok=True)

# Logging setup - for better error tracking
logging.basicConfig(level=logging.ERROR)  # Set logging level to ERROR

# Initialize components
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

llm = ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=0.3,
    num_gpu=1,
    num_ctx=2048,
    base_url='http://localhost:11434'
)

# --- Session state ---
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'active_db' not in st.session_state:
    st.session_state.active_db = None

# --- Page config ---
st.set_page_config(page_title="DocMind", layout="wide")
pages = ["📤 Upload Documents", "💬 Chat with Documents", "📝 Generate Questions", "🗄️ Manage Databases"]  # Added "Manage Databases" page
page = st.sidebar.selectbox("Navigation", pages)

# # --- UI Styling (using Streamlit theme) ---
# st.markdown("""
#     <style>
#         [data-testid="stAppViewContainer"] {
#             background-color: #f0f2f6;
#         }
#         [data-testid="stHeader"] {
#             background-color: rgba(0,0,0,0);
#         }
#         [data-testid="stSidebar"] {
#             background-color: #e0e7ef;
#         }
#     </style>
# """, unsafe_allow_html=True)


# --- Utility Functions ---
def create_pdf(content):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 12)
    for line in content.split('\n'):
        text.textLine(line)
    c.drawText(text)
    c.save()
    buffer.seek(0)
    return buffer

def get_available_dbs():
    return [d for d in os.listdir(PERSIST_ROOT) if os.path.isdir(os.path.join(PERSIST_ROOT, d))]

def is_valid_db_name(db_name):
    # Basic validation: alphanumeric and hyphens, no spaces, not empty
    return bool(db_name) and db_name.replace("-", "").isalnum() and " " not in db_name

def process_pdfs(uploaded_files, db_name):
    if not is_valid_db_name(db_name):
        st.error("Invalid database name. Use alphanumeric characters and hyphens only, no spaces.")
        return None  # Indicate failure

    persist_dir = os.path.join(PERSIST_ROOT, db_name)
    if os.path.exists(persist_dir):
        st.error(f"Database '{db_name}' already exists. Please choose a different name.")
        return None  # Indicate failure

    documents = []
    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            loader = PyPDFLoader(temp_path)
            docs = loader.load_and_split()
            documents.extend(docs)
            os.unlink(temp_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_dir,
            client_settings=Settings(anonymized_telemetry=False)
        )
        # vectorstore.persist() # Removed persist() call - not needed anymore
        st.success(f"Created database '{db_name}' with {len(uploaded_files)} files!")  # Success message here, before return
        return vectorstore
    except Exception as e:
        logging.error(f"Error processing PDFs and creating database: {e}")  # Log detailed error
        st.error(f"Failed to create database '{db_name}'. Please check logs for details.")  # User friendly error
        return None  # Indicate failure

def delete_database(db_name):
    persist_dir = os.path.join(PERSIST_ROOT, db_name)
    try:
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
            st.success(f"Database '{db_name}' deleted successfully!")
            return True  # Indicate success
        else:
            st.warning(f"Database '{db_name}' not found.")
            return False  # Indicate failure
    except Exception as e:
        logging.error(f"Error deleting database '{db_name}': {e}")
        st.error(f"Failed to delete database '{db_name}'. Check logs for details.")
        return False  # Indicate failure

def rename_database(old_db_name, new_db_name):
    if not is_valid_db_name(new_db_name):
        st.error("Invalid new database name. Use alphanumeric characters and hyphens only, no spaces.")
        return False

    old_persist_dir = os.path.join(PERSIST_ROOT, old_db_name)
    new_persist_dir = os.path.join(PERSIST_ROOT, new_db_name)

    if not os.path.exists(old_persist_dir):
        st.error(f"Database '{old_db_name}' not found.")
        return False

    if os.path.exists(new_persist_dir):
        st.error(f"Database '{new_db_name}' already exists. Choose a different name.")
        return False

    try:
        os.rename(old_persist_dir, new_persist_dir)
        st.session_state.active_db = new_db_name if st.session_state.active_db == old_db_name else st.session_state.active_db  # Update active_db if renamed
        st.success(f"Database '{old_db_name}' renamed to '{new_db_name}' successfully!")
        return True
    except Exception as e:
        logging.error(f"Error renaming database '{old_db_name}' to '{new_db_name}': {e}")
        st.error(f"Failed to rename database. Check logs for details.")
        return False


# --- Page 1: Upload Documents ---
if page == pages[0]:
    st.title("📤 Document Upload Center")

    with st.form("upload_form"):
        db_name = st.text_input("Database Name (e.g., Physics-101)", value=f"MyDocs-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}",
                                 help="Name your knowledge base. Alphanumeric and hyphens only.")  # Help text for DB name
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                                  help="Upload one or more PDF documents to create your knowledge base.")  # Help text for file uploader
        submitted = st.form_submit_button("Create Knowledge Base")

        if submitted and files:
            if not is_valid_db_name(db_name):  # Double check validation before processing
                st.error("Invalid database name. Use alphanumeric characters and hyphens only, no spaces.")
            else:
                with st.spinner("Creating knowledge base..."):
                    if process_pdfs(files, db_name):  # Process and check for success
                        pass  # Success message is already in process_pdfs


elif page == pages[1]:
    st.title("💬 Document Chat Interface")

    available_dbs = get_available_dbs()
    if not available_dbs:
        st.warning("No databases found! Upload documents first in 'Upload Documents' page.")
        st.stop()

    selected_db = st.selectbox("Select Knowledge Base", available_dbs,
                                    help="Choose the knowledge base you want to chat with.")

    if selected_db != st.session_state.active_db:
        st.session_state.active_db = selected_db
        persist_dir = os.path.join(PERSIST_ROOT, selected_db)
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False))
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Refined Prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history, rephrase the new user question to be self-contained and relevant to the conversation. Focus on maintaining context from previous turns. If the question is already self-contained, return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant answering questions based on the context provided in the documents. Use only the information from the following context to answer the question: {context}. If the context does not contain the answer, truthfully say 'I cannot answer this question from the given documents.'. Be concise and to the point."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        # Wrap retrieval chain with message history
        st.session_state.rag_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history=lambda session_id: ChatMessageHistory(
                messages=[
                    HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                    for msg in st.session_state.messages
                ]
            ),
            input_messages_key="input",
            history_messages_key="chat_history",
        )


        if 'messages' not in st.session_state:
            st.session_state.messages = []

    # Chat interface
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about the documents in your knowledge base:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            full_response = ""  # Initialize an empty string to accumulate the response
            message_placeholder = st.empty()  # Create a placeholder in the UI

            try:
                for chunk in st.session_state.rag_chain.stream( # Use .stream instead of .invoke
                    {"input": prompt},
                    config={"configurable": {"session_id": "default"}}
                ):
                    if 'answer' in chunk: # Check if 'answer' key exists in the chunk
                        answer_part = chunk['answer']
                        full_response += answer_part
                        message_placeholder.markdown(full_response + "▌") # Display partial response with a cursor

                message_placeholder.markdown(full_response) # Final response without cursor
                st.session_state.messages.append({"role": "assistant", "content": full_response}) # Store full response

            except Exception as e:
                logging.error(f"Error during RAG chain streaming invocation: {e}")
                st.error(f"Sorry, I encountered an error while processing your query. Please try again or check logs.")
# --- Page 3: Question Generator ---
elif page == pages[2]:
    st.title("📝 Automated Question Generator")

    available_dbs = get_available_dbs()
    if not available_dbs:
        st.warning("No databases found! Upload documents first.")
        st.stop()

    selected_db = st.selectbox("Select Knowledge Base", available_dbs, help="Choose the knowledge base to generate questions from.")

    with st.form("question_form"):
        col1, col2 = st.columns(2)
        with col1:
            paper_type = st.radio("Type", ["MCQ", "Descriptive"], help="Choose the type of exam paper.")
            exam_type = st.selectbox("Exam", ["Midterm", "Final"], help="Select the exam type (Midterm or Final).")
        with col2:
            if paper_type == "Descriptive":
                # Marks distribution using a dynamic number input
                num_questions_slider = st.slider("Number of Questions", 5, 30, 10, help="Number of questions to generate.")
                marks_distribution = []
                st.write("Marks Distribution:")  # Label for dynamic inputs
                for i in range(num_questions_slider):
                    marks = st.number_input(f"Q{i+1} Marks", min_value=1, max_value=20, value=5, step=1, key=f"marks_{i}",
                                             help=f"Marks for Question {i+1}")  # Help text for each question mark
                    marks_distribution.append(marks)
                marks_str = ",".join(map(str, marks_distribution))  # Convert marks list to comma separated string for prompt

            else:
                marks_str = '1 each'  # Default for MCQ
                num_questions = st.slider("Number of Questions", 5, 30, 10, help="Number of MCQ questions.")
            num_questions = num_questions_slider if paper_type == 'Descriptive' else num_questions  # Use slider value for descriptive, slider already used for MCQ

        instructions = st.text_area("Special Instructions", help="Any special instructions for the exam paper (optional).")  # Help text for instructions
        submitted = st.form_submit_button("Generate Paper")

    if submitted:
        persist_dir = os.path.join(PERSIST_ROOT, selected_db)
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            client_settings=Settings(anonymized_telemetry=False))

        if paper_type == 'MCQ':
            prompt = f"""
                Generate a {exam_type} MCQ paper based on the documents in the knowledge base.
                - Number of questions: {num_questions}
                - Marks per question: 1
                - Instructions: {instructions}
                - For each question, generate 4 options, with only one correct answer.
                - Clearly indicate the correct answer for each question (e.g., with '(Correct Answer)').
                - Ensure the questions are relevant to the content of the documents and test understanding, not just factual recall.
                - Create plausible but incorrect distractors (wrong options) for each MCQ.
                Paper should be formatted clearly for students.
                Start with a title indicating the exam type and database name.
                """
        elif paper_type == 'Descriptive':
            prompt = f"""
                Generate a {exam_type} descriptive paper based on the documents in the knowledge base.
                - Number of questions: {num_questions}
                - Marks distribution: {marks_str} (Marks for each question respectively)
                - Instructions: {instructions}
                - Generate descriptive questions that require detailed answers, analysis, and critical thinking based on the documents.
                - Focus on questions that assess deeper understanding, not just simple factual recall.
                - Marks for each question should be clearly indicated next to each question.
                Paper should be formatted as a proper exam paper for students, with clear numbering and spacing.
                Start with a title indicating the exam type and database name.
                """
        else:
            st.error("Invalid paper type selected.")
            st.stop()

        try:
            with st.spinner("Generating exam paper..."):
                response = llm.invoke(prompt)
                pdf_buffer = create_pdf(response.content)
                st.download_button(
                    "Download Paper",
                    data=pdf_buffer,
                    file_name=f"{selected_db}-{exam_type}.pdf",
                    mime="application/pdf",
                    help="Download the generated exam paper in PDF format."  # Help text for download button
                )
                st.subheader("Preview of Generated Paper:")  # More informative subheader
                st.write(response.content)
        except Exception as e:
            logging.error(f"Error during question paper generation: {e}")
            st.error(f"Exam paper generation failed. Please check logs for details.")


# --- Page 4: Manage Databases ---
elif page == pages[3]:
    st.title("🗄️ Manage Knowledge Bases")
    available_dbs = get_available_dbs()

    if not available_dbs:
        st.info("No databases created yet. Go to 'Upload Documents' to create one.")
    else:
        st.subheader("Available Databases")
        for db_name in available_dbs:
            col1, col2, col3 = st.columns([3, 1, 1])  # Adjust column widths

            with col1:
                st.write(f"**{db_name}**")  # Bold database name
            with col2:
                if st.button("Rename", key=f"rename_{db_name}"):
                    st.session_state['rename_db'] = db_name  # Store db name to rename
                    st.session_state['show_rename_input'] = True  # Show input field
            with col3:
                if st.button("Delete", key=f"delete_{db_name}"):
                    if st.session_state.active_db == db_name:  # Deselect if deleting active DB
                        st.session_state.active_db = None
                    if delete_database(db_name):
                        available_dbs = get_available_dbs()  # Refresh DB list

        if 'show_rename_input' in st.session_state and st.session_state['show_rename_input']:
            db_to_rename = st.session_state['rename_db']
            new_db_name_input = st.text_input(f"New name for '{db_to_rename}'", value=db_to_rename)  # Pre-fill with current name
            if st.button("Confirm Rename", key="confirm_rename"):
                if rename_database(db_to_rename, new_db_name_input):
                    st.session_state['show_rename_input'] = False  # Hide input after rename
                    st.session_state['rename_db'] = None
                else:
                    pass  # Error message already shown by rename_database
                available_dbs = get_available_dbs()  # Refresh DB list after rename
                st.rerun()  # Refresh the page to update the list

        if st.button("Cancel Rename", key="cancel_rename"):
            st.session_state['show_rename_input'] = False  # Hide input
            st.session_state['rename_db'] = None