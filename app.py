import hashlib
import html
import time

import streamlit as st

from rag.loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import create_embeddings, model
from rag.vectorstore import create_vectorstore, search

from rag.llm import (
    stream_answer,
    generate_summary,
    translate_document,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 80% 0%,
                rgba(124, 58, 237, 0.08),
                transparent 30%
            ),
            #0a0e17;
        color: #f8fafc;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    /* Remove excessive Streamlit spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.65rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        background: #0c111c;
        border-right: 1px solid #20293a;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1.2rem;
    }


    /* =====================================================
       BRAND
       ===================================================== */

    .brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    .brand-icon {
        color: #8b5cf6;
    }

    .brand-subtitle {
        color: #7f8ba3;
        font-size: 12px;
        margin-top: 3px;
        margin-bottom: 22px;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        padding: 22px 0 12px;
    }

    .hero-title {
        font-size: clamp(34px, 5vw, 48px);
        font-weight: 850;
        letter-spacing: -2.5px;
        line-height: 1.05;
    }

    .hero-gradient {
        background: linear-gradient(
            90deg,
            #a78bfa,
            #8b5cf6,
            #c084fc
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        max-width: 720px;
        margin-top: 12px;
        color: #8994a8;
        font-size: 15px;
        line-height: 1.65;
    }


    /* =====================================================
       DOCUMENT CARD
       ===================================================== */

    .document-card {
        position: relative;
        overflow: hidden;

        background:
            linear-gradient(
                135deg,
                rgba(24, 32, 49, 0.98),
                rgba(14, 21, 34, 0.98)
            );

        border: 1px solid #293449;
        border-radius: 18px;
        padding: 20px;
        margin: 22px 0 18px;
    }

    .document-card::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(
            #8b5cf6,
            #6366f1
        );
    }

    .document-title {
        font-size: 17px;
        font-weight: 700;
        color: #f1f5f9;
        padding-left: 5px;
        word-break: break-word;
    }

    .document-meta {
        color: #7f8ba3;
        font-size: 13px;
        margin-top: 6px;
        padding-left: 5px;
    }

    .ready {
        display: inline-block;
        background: #0d2a1b;
        border: 1px solid #185c37;
        color: #4ade80;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: 12px;
        margin-top: 12px;
        margin-left: 5px;
    }


    /* =====================================================
       SECTION TITLE
       ===================================================== */

    .section-title {
        font-size: 17px;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 25px;
        margin-bottom: 10px;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        width: 100%;
        min-height: 44px;

        border-radius: 11px;
        border: 1px solid #293449;

        background: #131a28;
        color: #e5e7eb;

        font-size: 14px;
        font-weight: 600;

        transition:
            border-color 0.2s ease,
            background 0.2s ease,
            transform 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #8b5cf6;
        background: #1a2132;
        color: white;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }


    /* =====================================================
       PRIMARY BUTTON
       ===================================================== */

    .primary-btn .stButton > button {
        background: linear-gradient(
            135deg,
            #7c3aed,
            #8b5cf6
        );

        border: 1px solid #8b5cf6;
        color: white;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: #111827;
        border: 1px dashed #364156;
        border-radius: 14px;
        padding: 10px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6;
    }


    /* =====================================================
       CHAT
       ===================================================== */

    [data-testid="stChatMessage"] {
        background: transparent;
        padding-top: 7px;
        padding-bottom: 7px;
    }

    [data-testid="stChatMessageContent"] {
        line-height: 1.7;
    }

    [data-testid="stChatInput"] {
        border-radius: 15px;
    }


    /* =====================================================
       METRICS
       ===================================================== */

    .metric-card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid #242e40;
        border-radius: 14px;
        padding: 15px 12px;
        text-align: center;
        transition: border-color 0.2s ease;
    }

    .metric-card:hover {
        border-color: #3a465c;
    }

    .metric-value {
        font-size: 19px;
        font-weight: 750;
        color: #f1f5f9;
    }

    .metric-label {
        color: #7f8ba3;
        font-size: 10px;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }


    /* =====================================================
       SOURCE
       ===================================================== */

    .source {
        background: #101722;
        border-left: 3px solid #8b5cf6;
        padding: 9px 12px;
        border-radius: 7px;
        color: #9aa5b8;
        font-size: 12px;
        margin-top: 10px;
    }


    /* =====================================================
       EMPTY STATE
       ===================================================== */

    .empty-state {
        text-align: center;
        padding: 80px 20px 55px;
    }

    .empty-icon {
        font-size: 62px;
        margin-bottom: 15px;
    }

    .empty-title {
        font-size: 26px;
        font-weight: 750;
        color: #e5e7eb;
    }

    .empty-description {
        color: #7f8ba3;
        max-width: 600px;
        margin: 11px auto;
        font-size: 14px;
        line-height: 1.7;
    }


    /* =====================================================
       FEATURE CARDS
       ===================================================== */

    .feature-card {
        background: #111827;
        border: 1px solid #242e40;
        border-radius: 15px;
        padding: 18px;
        height: 100%;
    }

    .feature-icon {
        font-size: 23px;
        margin-bottom: 8px;
    }

    .feature-title {
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 5px;
    }

    .feature-description {
        color: #7f8ba3;
        font-size: 13px;
        line-height: 1.55;
    }


    /* =====================================================
       RESULT CARD
       ===================================================== */

    .result-header {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-bottom: 10px;
    }

    .result-icon {
        font-size: 20px;
    }

    .result-title {
        font-size: 18px;
        font-weight: 750;
        color: #f1f5f9;
    }


    /* =====================================================
       STATUS BADGE
       ===================================================== */

    .status-badge {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 11px;
        background: #172033;
        color: #a5b4fc;
        border: 1px solid #303c56;
    }


    /* =====================================================
       DIVIDER
       ===================================================== */

    .divider {
        height: 1px;
        background: #20293a;
        margin: 25px 0;
    }


    /* =====================================================
       MOBILE
       ===================================================== */

    @media (max-width: 768px) {

        .hero {
            padding-top: 10px;
        }

        .hero-title {
            font-size: 36px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "chunks": None,
    "index": None,
    "document_name": None,
    "document_hash": None,
    "chat_history": [],
    "action_result": None,
    "action_type": None,
    "last_question": None,
    "last_retrieved_count": 0,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clear_document():
    """Completely remove the current document."""

    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def clear_chat():
    """Clear only conversation history."""

    st.session_state.chat_history = []
    st.session_state.last_question = None
    st.session_state.last_retrieved_count = 0


def clear_action():
    """Clear summary / translation output."""

    st.session_state.action_result = None
    st.session_state.action_type = None


def get_file_hash(uploaded_file):
    """Create a stable hash to detect duplicate uploads."""

    data = uploaded_file.getvalue()

    return hashlib.sha256(data).hexdigest()


def safe_text(value):
    """Escape text before inserting it into HTML."""

    return html.escape(str(value))


def estimate_words(chunks):
    """Estimate document word count from chunks."""

    if not chunks:
        return 0

    return sum(
        len(chunk.split())
        for chunk in chunks
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    # -----------------------------------------------------
    # BRAND
    # -----------------------------------------------------

    st.markdown(
        """
        <div class="brand">
            <span class="brand-icon"></span>
            DocuMind AI
        </div>

        <div class="brand-subtitle">
            Intelligent document workspace
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------

    st.markdown("###  Document")

    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        help="Supported formats: PDF and TXT",
    )

    if uploaded_file:

        file_hash = get_file_hash(uploaded_file)

        st.info(
            f"📎 {uploaded_file.name}"
        )

        # -------------------------------------------------
        # DUPLICATE DETECTION
        # -------------------------------------------------

        already_processed = (
            st.session_state.document_hash == file_hash
            and st.session_state.index is not None
        )

        if already_processed:

            st.success(
                "This document is already processed."
            )

        else:

            st.markdown(
                '<div class="primary-btn">',
                unsafe_allow_html=True,
            )

            process_button = st.button(
                " Process Document",
                use_container_width=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

            if process_button:

                with st.status(
                    "Processing document...",
                    expanded=True,
                ) as status:

                    start_time = time.time()

                    try:

                        # =================================
                        # EXTRACTION
                        # =================================

                        status.write(
                            " Extracting document text..."
                        )

                        text = load_document(
                            uploaded_file
                        )

                        if not text or not text.strip():

                            raise ValueError(
                                "The document contains no readable text."
                            )

                        status.write(
                            f"✓ Extracted approximately "
                            f"{len(text.split()):,} words."
                        )

                        # =================================
                        # CHUNKING
                        # =================================

                        status.write(
                            " Splitting document into chunks..."
                        )

                        chunks = chunk_text(text)

                        if not chunks:

                            raise ValueError(
                                "No text chunks were created."
                            )

                        status.write(
                            f"✓ Created {len(chunks):,} chunks."
                        )

                        # =================================
                        # EMBEDDINGS
                        # =================================

                        status.write(
                            " Creating embeddings..."
                        )

                        embeddings = create_embeddings(
                            chunks
                        )

                        status.write(
                            "✓ Embeddings created."
                        )

                        # =================================
                        # VECTOR STORE
                        # =================================

                        status.write(
                            " Building FAISS vector index..."
                        )

                        index = create_vectorstore(
                            embeddings
                        )

                        # =================================
                        # SAVE STATE
                        # =================================

                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.document_name = (
                            uploaded_file.name
                        )
                        st.session_state.document_hash = (
                            file_hash
                        )

                        clear_chat()
                        clear_action()

                        elapsed = time.time() - start_time

                        status.update(
                            label=(
                                f"Document ready "
                                f"({elapsed:.1f}s)"
                            ),
                            state="complete",
                        )

                        st.rerun()

                    except Exception as e:

                        status.update(
                            label="Processing failed",
                            state="error",
                        )

                        st.error(
                            f"❌ {e}"
                        )

    # -----------------------------------------------------
    # DOCUMENT INFO
    # -----------------------------------------------------

    if st.session_state.index is not None:

        st.divider()

        st.markdown("###  Document Info")

        st.success("● Ready")

        st.caption(
            f" {st.session_state.document_name}"
        )

        st.caption(
            f" {len(st.session_state.chunks):,} chunks indexed"
        )

        st.caption(
            f" ~{estimate_words(st.session_state.chunks):,} words"
        )

        st.divider()

        # -------------------------------------------------
        # CHAT CONTROLS
        # -------------------------------------------------

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
        ):

            clear_chat()
            st.rerun()

        if st.button(
            "✕ Remove Document",
            use_container_width=True,
        ):

            clear_document()
            st.rerun()

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption(
        "DocuMind AI • RAG powered"
    )

    st.caption(
        "PDF / TXT • FAISS • Groq"
    )


# =========================================================



    # =====================================================
    # QUICK ACTIONS
    # =====================================================

    st.markdown(
        '<div class="section-title">⚡ Quick Actions</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        summary_button = st.button(
            " Summarize Document",
            use_container_width=True,
        )

    with col2:

        translate_button = st.button(
            "🇳🇵 Translate to Nepali",
            use_container_width=True,
        )

    with col3:

        clear_result_button = st.button(
            "↺ Clear Result",
            use_container_width=True,
        )

    # -----------------------------------------------------
    # CLEAR RESULT
    # -----------------------------------------------------

    if clear_result_button:

        clear_action()
        st.rerun()

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    if summary_button:

        with st.spinner(
            " Creating document summary..."
        ):

            try:

                context = "\n\n".join(
                    st.session_state.chunks
                )

                result = generate_summary(
                    context
                )

                st.session_state.action_result = result
                st.session_state.action_type = "summary"

            except Exception as e:

                st.error(
                    f"Summary failed: {e}"
                )

    # -----------------------------------------------------
    # TRANSLATION
    # -----------------------------------------------------

    if translate_button:

        with st.spinner(
            "🇳🇵 Translating document to Nepali..."
        ):

            try:

                result = translate_document(
                    st.session_state.chunks
                )

                st.session_state.action_result = result
                st.session_state.action_type = "translation"

            except Exception as e:

                st.error(
                    f"Translation failed: {e}"
                )


# =========================================================
# ACTION RESULT
# =========================================================

if st.session_state.action_result:

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.action_type == "summary":

        st.markdown(
            """
            <div class="result-header">
                <span class="result-icon"></span>
                <span class="result-title">
                    Document Summary
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="result-header">
                <span class="result-icon">🇳🇵</span>
                <span class="result-title">
                    Nepali Translation
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "Complete translated document"
        )

    st.markdown(
        st.session_state.action_result
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    file_name = (
        "summary.txt"
        if st.session_state.action_type == "summary"
        else "nepali_translation.txt"
    )

    st.download_button(
        label="⬇️ Download Result",
        data=st.session_state.action_result,
        file_name=file_name,
        mime="text/plain",
    )


# =========================================================
# CHAT
# =========================================================

if st.session_state.index is not None:

    st.markdown(
        '<div class="section-title">💬 Chat with your document</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Answers are generated using the most relevant "
        "sections retrieved from your document."
    )

    # -----------------------------------------------------
    # RETRIEVAL SETTINGS
    # -----------------------------------------------------

    with st.expander(
        "⚙️ Retrieval settings",
        expanded=False,
    ):

        retrieval_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help=(
                "More chunks provide more context but may "
                "increase prompt size."
            ),
        )

    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # CHAT INPUT
    # -----------------------------------------------------

    question = st.chat_input(
        "Ask anything about your document..."
    )


    if question:

        question = question.strip()

        if not question:
            st.warning(
                "Please enter a question."
            )
            st.stop()

        st.session_state.last_question = question

        # =================================================
        # USER MESSAGE
        # =================================================

        with st.chat_message("user"):

            st.markdown(
                question
            )

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # =================================================
        # QUESTION EMBEDDING
        # =================================================

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    " Searching your document..."
                ):

                    query_embedding = model.encode(
                        question,
                        convert_to_numpy=True,
                    )

                    relevant_chunks = search(
                        st.session_state.index,
                        query_embedding,
                        st.session_state.chunks,
                        k=retrieval_k,
                    )

                if not relevant_chunks:

                    answer = (
                        "I couldn't find relevant information "
                        "in the document for that question."
                    )

                    st.markdown(answer)

                else:

                    # =====================================
                    # CONTEXT
                    # =====================================

                    context = "\n\n".join(
                        relevant_chunks
                    )

                    # =====================================
                    # AI RESPONSE
                    # =====================================

                    answer = st.write_stream(
                        stream_answer(
                            question,
                            context,
                        )
                    )

                    # =====================================
                    # SAVE RESPONSE
                    # =====================================

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    st.session_state.last_retrieved_count = (
                        len(relevant_chunks)
                    )

                    # =====================================
                    # RETRIEVAL INFO
                    # =====================================

                    st.markdown(
                        f"""
                        <div class="source">
                             Answer generated from
                            <strong>
                                {len(relevant_chunks)}
                            </strong>
                            relevant document chunks.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as e:

                error_message = (
                    f"AI response failed: {e}"
                )

                st.error(
                    error_message
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                
            </div>

            <div class="empty-title">
                Start with a document
            </div>

            <div class="empty-description">

                Upload a PDF or TXT file from the sidebar
                to create your private AI document workspace.

                <br><br>

                Once indexed, you can ask questions,
                generate summaries, or translate the
                complete document into Nepali.

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # =====================================================
    # FEATURE CARDS
    # =====================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    
                </div>

                <div class="feature-title">
                    Ask Questions
                </div>

                <div class="feature-description">
                    Ask natural-language questions
                    and retrieve answers directly
                    from your document.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    
                </div>

                <div class="feature-title">
                    Summarize
                </div>

                <div class="feature-description">
                    Turn long documents into
                    concise, easy-to-understand
                    summaries.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🇳🇵
                </div>

                <div class="feature-title">
                    Nepali Translation
                </div>

                <div class="feature-description">
                    Translate the complete document
                    into Nepali with a single click.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )
