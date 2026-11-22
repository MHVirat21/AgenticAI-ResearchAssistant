
import io

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==========================================
# 1. INITIALIZE TRANSFORMER EMBEDDINGS
# ==========================================

def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


# ==========================================
# 2. EXTRACT TEXT FROM PDF
# ==========================================

def extract_pdf_text(pdf_bytes):

    reader = PdfReader(io.BytesIO(pdf_bytes))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():

            document = Document(
                page_content=text,
                metadata={
                    "page": page_number
                }
            )

            documents.append(document)

    if not documents:
        raise ValueError(
            "No readable text found in PDF. "
            "Scanned PDFs require OCR."
        )

    return documents, len(reader.pages)


# ==========================================
# 3. SPLIT TEXT INTO CHUNKS
# ==========================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks generated.")

    return chunks


# ==========================================
# 4. CREATE FAISS VECTOR DATABASE
# ==========================================

def create_vectorstore(chunks):

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# ==========================================
# 5. COMPLETE PDF PROCESSING PIPELINE
# ==========================================

def process_pdf(pdf_bytes):

    documents, total_pages = extract_pdf_text(pdf_bytes)

    chunks = split_documents(documents)

    vectorstore = create_vectorstore(chunks)

    return vectorstore, total_pages, len(chunks)


# ==========================================
# 6. RETRIEVE RELEVANT DOCUMENT CHUNKS
# ==========================================

def retrieve_documents(vectorstore, question, k=4):

    results = vectorstore.similarity_search(
        query=question,
        k=k
    )

    return results


# ==========================================
# 7. RAG QUESTION ANSWERING
# ==========================================

def ask_pdf_question(vectorstore, question, llm):

    relevant_documents = retrieve_documents(
        vectorstore,
        question,
        k=4
    )

    if not relevant_documents:

        return (
            "No relevant information was found in the document.",
            []
        )

    context = "\n\n".join(
        [
            f"Source Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"Content: {doc.page_content}"
            for doc in relevant_documents
        ]
    )

    prompt = f"""
You are an intelligent AI Research Assistant.

Answer the user's question using the provided document context.

RULES:

1. Give accurate and understandable answers.
2. Do not invent facts.
3. Use only the supplied document context.
4. If the answer is not available, say so clearly.
5. Explain complex concepts in simple language.
6. Include relevant source page numbers.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):

        answer = "\n".join(
            str(item.get("text", item))
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    source_pages = sorted(
        set(
            doc.metadata.get("page", "Unknown")
            for doc in relevant_documents
        ),
        key=lambda page: (isinstance(page, str), str(page))
    )

    return answer, source_pages


# ==========================================
# 8. DOCUMENT SUMMARY
# ==========================================

def summarize_document(vectorstore, llm):

    relevant_documents = vectorstore.similarity_search(
        "Main topics, objectives, important findings and conclusions",
        k=6
    )

    context = "\n\n".join(
        doc.page_content
        for doc in relevant_documents
    )

    prompt = f"""
You are an expert document summarization assistant.

Summarize the following document content.

Include:
1. Document overview
2. Important topics
3. Key findings
4. Important points
5. Conclusion

DOCUMENT CONTENT:

{context}

SUMMARY:
"""

    response = llm.invoke(prompt)

    return response.content

def generate_quiz(vectorstore, llm, num_questions=5):
    """
    Generate multiple-choice questions from an uploaded PDF.
    """

    import json
    import re

    # Retrieve relevant document content
    documents = vectorstore.similarity_search(
        "Important facts, concepts, technical skills, projects, "
        "definitions and key information in this document",
        k=8
    )

    if not documents:
        raise ValueError("No document content found.")

    # Combine retrieved content
    context = "\n\n".join(
        [
            f"Source Page {doc.metadata.get('page', 0) + 1}:\n"
            f"{doc.page_content}"
            for doc in documents
        ]
    )

    # Quiz generation prompt
    prompt = f"""
You are an expert educational quiz generator.

Generate exactly {num_questions} multiple-choice questions
using ONLY the information in the document below.

DOCUMENT:
{context}

INSTRUCTIONS:
1. Generate questions based only on the document.
2. Each question must have exactly four options.
3. Only one option must be correct.
4. Include the correct answer.
5. Provide a short explanation based on the document.
6. Do not invent information.

Return ONLY valid JSON in this exact format:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": "Option A",
            "explanation": "Explanation from the document"
        }}
    ]
}}
"""

    # Call the LLM
    response = llm.invoke(prompt)
    result = response.content

    # Remove Markdown code fences if present
    result = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        result.strip(),
        flags=re.IGNORECASE
    )

    # Parse JSON
    try:
        quiz_data = json.loads(result)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Could not parse quiz response as JSON: {error}"
        )

    questions = quiz_data.get("questions", [])

    if len(questions) != num_questions:
        raise ValueError(
            f"Expected {num_questions} questions, "
            f"but received {len(questions)}."
        )

    # Validate generated questions
    for index, question in enumerate(questions):

        if not all(
            key in question
            for key in [
                "question",
                "options",
                "answer",
                "explanation"
            ]
        ):
            raise ValueError(
                f"Question {index + 1} has missing fields."
            )

        if len(question["options"]) != 4:
            raise ValueError(
                f"Question {index + 1} must have four options."
            )

        if question["answer"] not in question["options"]:
            raise ValueError(
                f"Question {index + 1} has an invalid answer."
            )

    return questions

def extract_key_points(vectorstore, llm):
    """
    Extract important information from an uploaded PDF.
    """

    try:
        # Retrieve document chunks
        docs = vectorstore.similarity_search(
            "Extract all important information, facts, names, dates, skills, "
            "projects, results, definitions and conclusions from this document.",
            k=12
        )

        if not docs:
            return "No document content was found."

        # Combine retrieved text
        document_text = "\n\n".join(
            doc.page_content for doc in docs
        )

        prompt = f"""
You are an AI research assistant.

Analyze the following document and extract its important information.

DOCUMENT:
{document_text}

Prepare the output using these sections:

## 1. Main Topic
Explain what the document is mainly about.

## 2. Important Key Points
List the most important facts and information.

## 3. Names, Dates and Numbers
Extract important names, dates, percentages, measurements and figures.

## 4. Technical Skills and Concepts
List technical skills, tools, technologies and concepts if present.

## 5. Projects and Experience
Summarize projects, work experience or practical activities if present.

## 6. Important Findings
Explain the important results, achievements or conclusions.

## 7. Quick Revision Notes
Provide short bullet points useful for quick revision.

IMPORTANT RULES:
- Use only information found in the document.
- Do not invent information.
- Do not assume missing details.
- If a section is not covered, write "Not mentioned in the document."
- Preserve the original meaning.
- Use clear headings and bullet points.
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:
        return f"Key points extraction failed: {str(e)}"
