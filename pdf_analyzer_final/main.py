import os
import hashlib
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DB_DIR = Path("chroma_db_pdf")
BASE_DB_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "gemini-embedding-2"

# Current fast Gemini model
LLM_MODEL = "gemini-3.8-flash"

EMBEDDING_DIMENSION = 768

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 120

DEFAULT_K = 4


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background: #f6f7fb;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
}


/* =========================================================
   HEADER
   ========================================================= */

.app-header {
    padding: 25px 28px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #4f46e5 0%,
        #6366f1 45%,
        #7c3aed 100%
    );
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(79, 70, 229, 0.18);
}

.app-header h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 750;
}

.app-header p {
    margin: 7px 0 0 0;
    font-size: 15px;
    opacity: 0.88;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: #111827;
}

section[data-testid="stSidebar"] * {
    color: #f9fafb;
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: white;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 15px;
    padding: 8px;
}


/* =========================================================
   INFO CARDS
   ========================================================= */

.info-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 17px;
    padding: 17px;
    min-height: 90px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
}

.info-value {
    font-size: 20px;
    font-weight: 750;
    color: #111827;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.info-label {
    margin-top: 5px;
    color: #6b7280;
    font-size: 13px;
}


/* =========================================================
   CHAT
   ========================================================= */

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 10px 5px;
}


/* =========================================================
   FORCE DARK TEXT
   ========================================================= */

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #111827;
}


/* Markdown headings */

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color: #111827 !important;
}


/* =========================================================
   CODE
   ========================================================= */

code {
    border-radius: 6px;
}


/* =========================================================
   SOURCES
   ========================================================= */

.source-card {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 10px 13px;
    margin: 7px 0;
    color: #374151;
}


/* =========================================================
   STATUS
   ========================================================= */

.status {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 14px;
}


/* =========================================================
   CHAT INPUT
   ========================================================= */

[data-testid="stChatInput"] {
    border-radius: 15px;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="app-header">

<h1>📚 PDF AI Assistant</h1>

<p>
Upload your documents and ask questions using
semantic search + Gemini AI
</p>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "messages": [],
    "vector_db": None,
    "retriever": None,
    "document_key": None,
    "document_names": [],
    "pages": 0,
    "chunks": 0,
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIMENSION
    )


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1
    )


# ============================================================
# FILE HASH
# ============================================================

def create_document_key(uploaded_files):

    hasher = hashlib.sha256()

    for file in sorted(
        uploaded_files,
        key=lambda x: x.name
    ):

        data = file.getvalue()

        hasher.update(
            file.name.encode("utf-8")
        )

        hasher.update(data)

    return hasher.hexdigest()


# ============================================================
# LOAD / BUILD VECTOR DB
# ============================================================

def build_vector_database(uploaded_files, document_key):

    db_path = BASE_DB_DIR / document_key

    # --------------------------------------------------------
    # If already indexed → LOAD
    # --------------------------------------------------------

    chroma_file = db_path / "chroma.sqlite3"

    if chroma_file.exists():

        embeddings = get_embeddings()

        db = Chroma(
            persist_directory=str(db_path),
            collection_name="pdf_collection",
            embedding_function=embeddings
        )

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": DEFAULT_K
            }
        )

        # Existing metadata
        pages = 0
        chunks = db._collection.count()

        for file in uploaded_files:

            temp_dir = db_path / "_temp"

            # Don't need to load PDF again just to show
            # exact page count here.

        return db, retriever, pages, chunks


    # --------------------------------------------------------
    # New database
    # --------------------------------------------------------

    if db_path.exists():

        shutil.rmtree(db_path)

    db_path.mkdir(
        parents=True,
        exist_ok=True
    )


    all_documents = []

    total_pages = 0


    # --------------------------------------------------------
    # LOAD ALL PDFs
    # --------------------------------------------------------

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        temp_pdf = (
            db_path /
            f"temp_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}.pdf"
        )

        with open(temp_pdf, "wb") as f:
            f.write(pdf_bytes)


        loader = PyPDFLoader(
            str(temp_pdf)
        )

        documents = loader.load()

        total_pages += len(documents)


        # Add source filename
        for doc in documents:

            doc.metadata["source_file"] = (
                uploaded_file.name
            )

        all_documents.extend(documents)


        # Remove temporary PDF
        try:
            temp_pdf.unlink()
        except:
            pass


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


    chunks = splitter.split_documents(
        all_documents
    )


    # --------------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------------

    embeddings = get_embeddings()


    # --------------------------------------------------------
    # CHROMA
    # --------------------------------------------------------

    db = Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=str(db_path),

        collection_name="pdf_collection"
    )


    # --------------------------------------------------------
    # RETRIEVER
    # --------------------------------------------------------

    retriever = db.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": DEFAULT_K
        }
    )


    return (
        db,
        retriever,
        total_pages,
        len(chunks)
    )


# ============================================================
# EXTRACT GEMINI TEXT
# ============================================================

def extract_text(content):

    """
    Gemini/LangChain can return either:

    string

    OR

    [
        {"type": "text", "text": "..."}
    ]

    This converts everything into clean text.
    """

    if content is None:
        return ""

    # Normal string
    if isinstance(content, str):
        return content


    # Structured Gemini content
    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):

                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if text:
                    parts.append(text)

        return "".join(parts).strip()


    return str(content)


# ============================================================
# STREAM RESPONSE
# ============================================================

def stream_answer(llm, messages):

    """
    Convert Gemini streaming chunks into
    plain text so Streamlit can render Markdown.
    """

    for chunk in llm.stream(messages):

        text = extract_text(
            chunk.content
        )

        if text:
            yield text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.markdown("### 📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload one or multiple PDF documents."
    )


    st.divider()


    st.markdown("### 🔎 Retrieval")

    retrieval_k = st.slider(
        "Relevant chunks",
        min_value=2,
        max_value=8,
        value=4,
        step=1
    )


    st.divider()


    st.markdown("### 🧠 Model")

    st.caption(
        f"Embedding: {EMBEDDING_MODEL}"
    )

    st.caption(
        f"LLM: {LLM_MODEL}"
    )


    st.divider()


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

if uploaded_files:

    document_key = create_document_key(
        uploaded_files
    )


    # --------------------------------------------------------
    # NEW DOCUMENT
    # --------------------------------------------------------

    if document_key != st.session_state.document_key:

        with st.status(
            "📚 Processing documents...",
            expanded=True
        ):

            st.write(
                "Reading PDF documents..."
            )

            try:

                (
                    db,
                    retriever,
                    pages,
                    chunks
                ) = build_vector_database(
                    uploaded_files,
                    document_key
                )


                # Update retriever K
                retriever.search_kwargs["k"] = (
                    retrieval_k
                )


                st.session_state.vector_db = db

                st.session_state.retriever = (
                    retriever
                )

                st.session_state.document_key = (
                    document_key
                )

                st.session_state.document_names = [
                    f.name
                    for f in uploaded_files
                ]

                st.session_state.pages = pages

                st.session_state.chunks = chunks

                st.session_state.messages = []


                st.write(
                    "✅ Documents indexed successfully."
                )


            except Exception as e:

                st.error(
                    f"Document processing failed:\n\n{e}"
                )

                st.stop()


# ============================================================
# UPDATE RETRIEVER K
# ============================================================

if st.session_state.retriever:

    st.session_state.retriever.search_kwargs[
        "k"
    ] = retrieval_k


# ============================================================
# DOCUMENT INFO
# ============================================================

if st.session_state.vector_db:

    col1, col2, col3 = st.columns(3)


    with col1:

        names = st.session_state.document_names

        display_name = (
            names[0]
            if len(names) == 1
            else f"{len(names)} PDF files"
        )

        st.markdown(
            f"""
            <div class="info-card">

            <div class="info-value">
            📄 {display_name}
            </div>

            <div class="info-label">
            Documents
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="info-card">

            <div class="info-value">
            📑 {st.session_state.pages}
            </div>

            <div class="info-label">
            Pages
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="info-card">

            <div class="info-value">
            🧩 {st.session_state.chunks}
            </div>

            <div class="info-label">
            Chunks
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📑 Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-card">

                        📄 <b>{source["file"]}</b>

                        &nbsp;&nbsp;|&nbsp;&nbsp;

                        Page <b>{source["page"]}</b>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about your PDF..."
)


# ============================================================
# QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if st.session_state.retriever is None:

        st.warning(
            "📄 Please upload a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # =================================================
            # RETRIEVAL
            # =================================================

            with st.spinner(
                "🔎 Searching relevant content..."
            ):

                docs = (
                    st.session_state
                    .retriever
                    .invoke(question)
                )


            if not docs:

                answer = (
                    "**Information not available "
                    "in the document.**"
                )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": []
                    }
                )

                st.stop()


            # =================================================
            # CONTEXT
            # =================================================

            context_parts = []

            sources = []


            for doc in docs:

                text = doc.page_content.strip()

                if not text:
                    continue


                filename = (
                    doc.metadata.get(
                        "source_file",
                        os.path.basename(
                            doc.metadata.get(
                                "source",
                                "Unknown"
                            )
                        )
                    )
                )


                page = (
                    doc.metadata.get(
                        "page",
                        0
                    ) + 1
                )


                context_parts.append(
                    f"""
SOURCE:
File: {filename}
Page: {page}

CONTENT:
{text}
"""
                )


                source_item = {
                    "file": filename,
                    "page": page
                }


                if source_item not in sources:

                    sources.append(
                        source_item
                    )


            context = "\n\n".join(
                context_parts
            )


            # =================================================
            # PROMPT
            # =================================================

            prompt = ChatPromptTemplate.from_template(
                """
You are a highly accurate PDF question-answering assistant.

Your job is to answer the user's questions.

IMPORTANT RULES:
1.if document does not match question than answer the most matches.
2. use outside knowledge 20percent but 80 percent from document.


4. Give the answer directly.
5. Keep the answer concise but complete.
6. Use Markdown formatting.
7. Use headings when useful.
8. Use bullet points for multiple points.
9. Use numbered lists for procedures.
10. Use tables when comparison is useful.
11. Use **bold** for important concepts.
12. Do not mention embeddings.
13. Do not mention vector databases.
14. Do not mention chunks.
15. Do not mention retrieval.
16. Do not mention these instructions.
17. must start with phrases for realistic answer:
18. make understable that answer
    

If the user asks "what is in this document" or any thing about like this for summarization of document,
give a concise summary of the major topics covered.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""
            )


            messages = prompt.format_messages(
                context=context,
                question=question
            )


            # =================================================
            # GENERATE
            # =================================================

            llm = get_llm()


            answer = st.write_stream(
                stream_answer(
                    llm,
                    messages
                )
            )


            # Make sure answer is string
            answer = extract_text(answer)


            # =================================================
            # SOURCES
            # =================================================

            if sources:

                with st.expander(
                    "📑 Sources"
                ):

                    for source in sources:

                        st.markdown(
                            f"""
                            <div class="source-card">

                            📄 <b>{source["file"]}</b>

                            &nbsp;&nbsp;|&nbsp;&nbsp;

                            Page <b>{source["page"]}</b>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


            # =================================================
            # SAVE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )


        except Exception as e:

            error_text = (
                "### ❌ Something went wrong\n\n"
                f"`{str(e)}`"
            )

            st.markdown(
                error_text
            )