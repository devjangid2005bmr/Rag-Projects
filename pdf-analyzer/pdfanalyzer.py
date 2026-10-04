import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in .env file"
    )


# =========================================================
# GEMINI MODEL
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
)


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

class StudyAnalysis(BaseModel):

    title: str = Field(
        description="The title or main subject of the PDF"
    )

    overview: str = Field(
        description="A clear overall explanation of the PDF"
    )

    full_summary: str = Field(
        description="Detailed but easy-to-understand summary"
    )

    topics: List[str] = Field(
        description="Major topics covered in the document"
    )

    key_points: List[str] = Field(
        description="Most important points students should remember"
    )

    short_notes: List[str] = Field(
        description="Concise revision notes for important concepts"
    )

    definitions: List[str] = Field(
        description="Important definitions from the document"
    )

    formulas: List[str] = Field(
        description="Important formulas, equations or rules"
    )

    important_questions: List[str] = Field(
        description="Likely important questions based only on the PDF"
    )

    exam_tips: List[str] = Field(
        description="Useful exam-focused points based on the PDF"
    )

    quick_revision: List[str] = Field(
        description="Very short last-minute revision points"
    )


structured_model = model.with_structured_output(
    StudyAnalysis
)


# =========================================================
# LOAD PDF
# =========================================================

def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    return docs
 

# =========================================================
# SPLIT DOCUMENT
# =========================================================

def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=7000,
        chunk_overlap=700
    )

    chunks = splitter.split_documents(docs)

    return chunks


# =========================================================
# SUMMARIZE CHUNKS
# =========================================================

def summarize_chunks(chunks):

    summaries = []

    for i, chunk in enumerate(chunks):

        page_number = (
            chunk.metadata.get("page", 0) + 1
        )

        prompt = f"""
You are an expert university teacher and study-notes creator.

Analyze the following section of a student's PDF.

SECTION NUMBER:
{i + 1}

PAGE:
{page_number}

IMPORTANT RULES:

1. Use ONLY information present in the provided content.
2. Do not invent facts.
3. Preserve technical terminology.
4. Identify important concepts.
5. Identify definitions.
6. Identify formulas if present.
7. Identify examples if present.
8. Identify exam-important information.
9. Make the explanation student-friendly.
10. Do not unnecessarily repeat information.

PDF CONTENT:

{chunk.page_content}
"""

        response = model.invoke(prompt)

        summaries.append(
            f"""
PAGE {page_number}

{response.content}
"""
        )

    return summaries


# =========================================================
# FINAL STUDY ANALYSIS
# =========================================================

def generate_final_analysis(summaries):

    combined_text = "\n\n".join(summaries)

    # Safety limit for very large documents
    if len(combined_text) > 100000:
        combined_text = combined_text[:100000]

    prompt = f"""
You are DOC-DEV.AI, an expert academic study assistant.

Create a high-quality study guide from the PDF analysis below.

The user is a university student preparing for exams.

Your task is to create:

1. PDF title / subject
2. Overall overview
3. Full understandable summary
4. Major topics
5. Key points
6. Short notes
7. Important definitions
8. Important formulas
9. Important exam questions
10. Exam tips
11. Last-minute quick revision

STRICT RULES:

- Use ONLY the information contained in the supplied PDF analysis.
- Never invent information.
- Never add outside facts.
- Preserve the terminology used in the PDF.
- Merge repeated concepts intelligently.
- Remove unnecessary repetition.
- Explain difficult concepts clearly.
- Prioritize information that appears important for examinations.
- If formulas are present, write them accurately.
- If no formulas exist, return an empty list.
- If a category is not present, return an empty list.
- Important questions must be based on the actual PDF content.
- Make the final result useful for revision.

PDF ANALYSIS:

{combined_text}
"""

    result = structured_model.invoke(prompt)

    return result


# =========================================================
# MAIN PIPELINE
# =========================================================

def analyze_pdf(pdf_path):

    # 1. Load
    docs = load_pdf(pdf_path)

    if not docs:
        raise ValueError(
            "No content could be extracted from this PDF."
        )

    # 2. Check extracted text
    total_text = "".join(
        doc.page_content for doc in docs
    )

    if len(total_text.strip()) < 100:
        raise ValueError(
            "Very little text was extracted. "
            "This may be a scanned/image-only PDF."
        )

    # 3. Split
    chunks = split_documents(docs)

    # 4. Summarize chunks
    summaries = summarize_chunks(chunks)

    # 5. Final analysis
    analysis = generate_final_analysis(
        summaries
    )

    return {
        "pages": len(docs),
        "chunks": len(chunks),
        "analysis": analysis
    }