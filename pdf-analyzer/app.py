import streamlit as st
import tempfile
import os

from pdfanalyzer import analyze_pdf


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DOC-DEV.AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(99, 102, 241, 0.15),
            transparent 35%
        ),
        radial-gradient(
            circle at top right,
            rgba(168, 85, 247, 0.12),
            transparent 35%
        ),
        #080b12;
}


/* HEADER */

.hero {
    text-align: center;
    padding: 55px 20px 30px 20px;
}

.logo {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #8b5cf6,
        #6366f1,
        #06b6d4
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.tagline {
    font-size: 19px;
    color: #9ca3af;
    margin-top: 10px;
}

.badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 50px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(129,140,248,0.25);
    color: #a5b4fc;
    font-size: 13px;
    margin-bottom: 15px;
}


/* CARDS */

.card {
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(15px);
}

.card-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 7px;
}

.card-subtitle {
    color: #9ca3af;
    font-size: 14px;
}


/* STAT */

.stat {
    text-align: center;
    padding: 20px;
    background: rgba(255,255,255,0.035);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
}

.stat-number {
    font-size: 28px;
    font-weight: 800;
    color: #a78bfa;
}

.stat-label {
    color: #9ca3af;
    font-size: 13px;
}


/* SECTION */

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 15px;
}


/* INFO */

.info-box {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366f1;
    border-radius: 12px;
    padding: 18px;
    margin: 10px 0;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 13px;
    font-weight: 700;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: #0d111b;
    border-right: 1px solid rgba(255,255,255,0.06);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">

<div class="badge">
✦ AI-POWERED STUDY ASSISTANT
</div>

<div class="logo">
📚 DOC-DEV.AI
</div>

<div class="tagline">
Turn any academic PDF into powerful study notes.
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📚 DOC-DEV.AI")

    st.markdown(
        "Your intelligent PDF study assistant."
    )

    st.divider()

    st.markdown("### ✨ What you get")

    st.markdown("""
    - 📖 Full Summary
    - 🧠 Key Concepts
    - 📝 Short Notes
    - 📌 Definitions
    - 📐 Formulas
    - ❓ Important Questions
    - 🎯 Exam Tips
    - ⚡ Quick Revision
    """)

    st.divider()

    st.caption(
        "Powered by LangChain + Gemini"
    )


# =========================================================
# UPLOAD AREA
# =========================================================

st.markdown("""
<div class="card">

<div class="card-title">
📄 Upload your study material
</div>

<div class="card-subtitle">
Upload lecture notes, university syllabus, textbooks,
research papers or any text-based PDF.
</div>

</div>
""", unsafe_allow_html=True)


uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed"
)


# =========================================================
# ANALYZE
# =========================================================

if uploaded_file:

    st.success(
        f"✓ {uploaded_file.name} uploaded successfully"
    )

    col1, col2 = st.columns([3, 1])

    with col1:

        analyze_button = st.button(
            "🚀 Analyze My PDF",
            type="primary"
        )

    with col2:

        if st.button("🗑️ Clear"):
            st.rerun()


    if analyze_button:

        with st.spinner(
            "🧠 DOC-DEV.AI is reading your PDF..."
        ):

            temp_path = None

            try:

                # Temporary PDF
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = tmp.name


                # Analyze
                result = analyze_pdf(
                    temp_path
                )

                analysis = result["analysis"]


                # =================================================
                # STATS
                # =================================================

                st.markdown(
                    '<div class="section-title">📊 Document Overview</div>',
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown(
                        f"""
                        <div class="stat">
                            <div class="stat-number">
                                {result["pages"]}
                            </div>
                            <div class="stat-label">
                                Pages
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:
                    st.markdown(
                        f"""
                        <div class="stat">
                            <div class="stat-number">
                                {result["chunks"]}
                            </div>
                            <div class="stat-label">
                                AI Sections
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c3:
                    st.markdown(
                        """
                        <div class="stat">
                            <div class="stat-number">
                                AI
                            </div>
                            <div class="stat-label">
                                Analysis
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # TITLE
                # =================================================

                st.markdown(
                    f"""
                    <div class="card">

                    <div class="card-title">
                    📘 {analysis.title}
                    </div>

                    <div class="info-box">
                    {analysis.overview}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # =================================================
                # SUMMARY
                # =================================================

                st.markdown(
                    '<div class="section-title">📖 Full Summary</div>',
                    unsafe_allow_html=True
                )

                with st.container(border=True):
                    st.write(
                        analysis.full_summary
                    )


                # =================================================
                # TOPICS
                # =================================================

                st.markdown(
                    '<div class="section-title">🧠 Major Topics</div>',
                    unsafe_allow_html=True
                )

                cols = st.columns(2)

                for i, topic in enumerate(
                    analysis.topics
                ):

                    with cols[i % 2]:

                        st.markdown(
                            f"""
                            <div class="info-box">
                            🔹 {topic}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


                # =================================================
                # KEY POINTS
                # =================================================

                st.markdown(
                    '<div class="section-title">⭐ Key Points</div>',
                    unsafe_allow_html=True
                )

                for point in analysis.key_points:

                    st.markdown(
                        f"✓ {point}"
                    )


                # =================================================
                # SHORT NOTES
                # =================================================

                st.markdown(
                    '<div class="section-title">📝 Short Notes</div>',
                    unsafe_allow_html=True
                )

                with st.expander(
                    "Open Short Notes",
                    expanded=True
                ):

                    for note in analysis.short_notes:

                        st.markdown(
                            f"• {note}"
                        )


                # =================================================
                # DEFINITIONS
                # =================================================

                st.markdown(
                    '<div class="section-title">📌 Important Definitions</div>',
                    unsafe_allow_html=True
                )

                with st.expander(
                    "View Definitions"
                ):

                    for definition in analysis.definitions:

                        st.markdown(
                            f"**{definition}**"
                        )


                # =================================================
                # FORMULAS
                # =================================================

                if analysis.formulas:

                    st.markdown(
                        '<div class="section-title">📐 Formulas & Rules</div>',
                        unsafe_allow_html=True
                    )

                    for formula in analysis.formulas:

                        st.code(
                            formula
                        )


                # =================================================
                # QUESTIONS
                # =================================================

                st.markdown(
                    '<div class="section-title">❓ Important Exam Questions</div>',
                    unsafe_allow_html=True
                )

                with st.expander(
                    "🎯 Questions you should prepare",
                    expanded=True
                ):

                    for i, question in enumerate(
                        analysis.important_questions,
                        1
                    ):

                        st.markdown(
                            f"**{i}. {question}**"
                        )


                # =================================================
                # EXAM TIPS
                # =================================================

                st.markdown(
                    '<div class="section-title">🎯 Exam Strategy</div>',
                    unsafe_allow_html=True
                )

                for tip in analysis.exam_tips:

                    st.markdown(
                        f"""
                        <div class="info-box">
                        💡 {tip}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                # =================================================
                # QUICK REVISION
                # =================================================

                st.markdown(
                    '<div class="section-title">⚡ Last-Minute Revision</div>',
                    unsafe_allow_html=True
                )

                for point in analysis.quick_revision:

                    st.markdown(
                        f"🔹 **{point}**"
                    )


                # =================================================
                # DOWNLOAD
                # =================================================

                st.divider()

                download_text = f"""
DOC-DEV.AI
===========

{analysis.title}

OVERVIEW
--------
{analysis.overview}

FULL SUMMARY
------------
{analysis.full_summary}

MAJOR TOPICS
------------
{chr(10).join("- " + x for x in analysis.topics)}

KEY POINTS
----------
{chr(10).join("- " + x for x in analysis.key_points)}

SHORT NOTES
-----------
{chr(10).join("- " + x for x in analysis.short_notes)}

DEFINITIONS
-----------
{chr(10).join("- " + x for x in analysis.definitions)}

FORMULAS
--------
{chr(10).join("- " + x for x in analysis.formulas)}

IMPORTANT QUESTIONS
-------------------
{chr(10).join("- " + x for x in analysis.important_questions)}

EXAM TIPS
---------
{chr(10).join("- " + x for x in analysis.exam_tips)}

QUICK REVISION
--------------
{chr(10).join("- " + x for x in analysis.quick_revision)}
"""

                st.download_button(
                    "⬇️ Download Study Notes",
                    download_text,
                    file_name="DOC-DEV-AI_Study_Notes.txt",
                    mime="text/plain"
                )


            except Exception as e:

                st.error(
                    f"Something went wrong: {str(e)}"
                )

            finally:

                if temp_path and os.path.exists(
                    temp_path
                ):
                    os.remove(temp_path)