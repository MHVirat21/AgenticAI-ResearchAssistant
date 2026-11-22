
import streamlit as st

from config import get_llm

from rag_pipeline import (
    process_pdf,
    ask_pdf_question,
    summarize_document,
    generate_quiz,
    extract_key_points
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Agentic AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM STYLING
# ==================================================

st.markdown("""
<style>

.main-header {
    font-size: 2.2rem;
    font-weight: 750;
    margin-bottom: 5px;
}

.sub-header {
    color: #8b949e;
    font-size: 1rem;
    margin-bottom: 20px;
}

div[data-testid="stMetric"] {
    background-color: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.2);
    padding: 15px;
    border-radius: 12px;
}

div[data-testid="stChatMessage"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SESSION STATE
# ==================================================

defaults = {
    "messages": [],
    "vectorstore": None,
    "pdf_name": None,
    "pdf_pages": 0,
    "pdf_chunks": 0,

    "quiz_questions": [],
    "quiz_submitted": False,
    "quiz_score": 0,
    "quiz_answers": {},
    "quiz_version": 0,

    "summary": None,
    "key_points": None,

    # Controls PDF uploader reset
    "uploader_version": 0
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def clear_generated_results():

    # Clear summary and key points
    st.session_state.summary = None
    st.session_state.key_points = None

    # Clear quiz data
    st.session_state.quiz_questions = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_answers = {}

    # Change quiz form key
    st.session_state.quiz_version += 1


def reset_document():

    # Clear processed document
    st.session_state.vectorstore = None
    st.session_state.pdf_name = None
    st.session_state.pdf_pages = 0
    st.session_state.pdf_chunks = 0

    # Clear chat history
    st.session_state.messages = []

    # Clear generated results
    clear_generated_results()

    # Create a new PDF uploader
    # This removes the previously selected PDF
    st.session_state.uploader_version += 1


def get_question_count():

    return sum(
        1
        for message in st.session_state.messages
        if message["role"] == "user"
    )


# ==================================================
# 5. APPLICATION HEADER
# ==================================================

st.markdown(
    '<div class="main-header">🤖 Agentic AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'Your intelligent assistant for PDF research, document analysis and learning.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# 6. SIDEBAR - DOCUMENT UPLOAD
# ==================================================

with st.sidebar:

    st.header("📁 Document Upload")

    # Dynamic key allows uploader to reset safely
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.uploader_version}",
        help="Upload a text-based PDF document for AI analysis."
    )

    if uploaded_file is not None:

        st.caption(f"Selected: {uploaded_file.name}")

        if st.button(
            "⚙️ Process PDF",
            use_container_width=True
        ):

            with st.spinner("Processing your PDF..."):

                try:

                    vectorstore, pages, chunks = process_pdf(
                        uploaded_file.getvalue()
                    )

                    # Save processed document
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_pages = pages
                    st.session_state.pdf_chunks = chunks

                    # Clear previous chat
                    st.session_state.messages = []

                    # Clear previous analysis results
                    clear_generated_results()

                    st.success("PDF processed successfully!")

                    st.rerun()

                except Exception as e:

                    st.error(f"PDF processing failed: {e}")

    # Document information
    if st.session_state.vectorstore is not None:

        st.divider()

        st.subheader("📄 Document Information")

        st.write(
            f"**File:** {st.session_state.pdf_name}"
        )

        st.write(
            f"**Pages:** {st.session_state.pdf_pages}"
        )

        st.write(
            f"**Chunks:** {st.session_state.pdf_chunks}"
        )

        st.success("Document Ready")

        # Reset document button
        if st.button(
            "🗑️ Reset Document",
            use_container_width=True
        ):

            reset_document()

            # Rerun creates a new empty PDF uploader
            st.rerun()

    st.divider()

    # Clear chat button
    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.caption("Powered by Groq, LangChain and FAISS")


# ==================================================
# 7. DASHBOARD METRICS
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "AI Engine",
        "Groq"
    )

with col2:

    st.metric(
        "Document Status",
        "Ready" if st.session_state.vectorstore is not None
        else "Not Uploaded"
    )

with col3:

    st.metric(
        "Questions Asked",
        get_question_count()
    )


st.divider()


# ==================================================
# 8. APPLICATION TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 AI Research Chat",
    "📑 Document Summary",
    "📝 Quiz Generator",
    "🔍 Key Points Extractor"
])


# ==================================================
# 9. TAB 1 - AI RESEARCH CHAT
# ==================================================

with tab1:

    st.header("💬 AI Research Chat")

    st.write(
        "Ask questions about your uploaded PDF and get AI-powered answers."
    )

    # Chat history area
    with st.container(height=450, border=True):

        if not st.session_state.messages:

            st.info(
                "👋 Welcome! Upload a PDF and ask your first research question."
            )

        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

    # Question form below chat history
    with st.form(
        key="research_question_form",
        clear_on_submit=True
    ):

        user_question = st.text_input(
            "Ask your research question",
            placeholder="Example: List all technical skills in my resume...",
            label_visibility="collapsed"
        )

        submitted = st.form_submit_button(
            "🔍 Ask Question",
            use_container_width=True
        )

    if submitted:

        if not user_question.strip():

            st.warning("Please enter a question.")

        elif st.session_state.vectorstore is None:

            st.warning("Please upload and process a PDF first.")

        else:

            try:

                with st.spinner("AI is researching your question..."):

                    llm = get_llm()

                    answer, source_pages = ask_pdf_question(
                        st.session_state.vectorstore,
                        user_question.strip(),
                        llm
                    )

                    if source_pages:

                        answer += (
                            "\n\n**Source Pages:** "
                            + ", ".join(
                                map(str, source_pages)
                            )
                        )

                # Save messages after successful response
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_question.strip()
                })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to answer your question: {e}"
                )


# ==================================================
# 10. TAB 2 - DOCUMENT SUMMARY
# ==================================================

with tab2:

    st.header("📑 AI Document Summary")

    st.write(
        "Generate an AI-powered summary of your uploaded document."
    )

    if st.session_state.vectorstore is None:

        st.info("Please upload and process a PDF first.")

    else:

        st.success(
            f"Document loaded: {st.session_state.pdf_name}"
        )

        if st.button(
            "✨ Generate Summary",
            use_container_width=True
        ):

            with st.spinner("Generating document summary..."):

                try:

                    llm = get_llm()

                    summary = summarize_document(
                        st.session_state.vectorstore,
                        llm
                    )

                    st.session_state.summary = summary

                except Exception as e:

                    st.error(
                        f"Summary generation failed: {e}"
                    )

        if st.session_state.summary:

            st.divider()

            st.subheader("📖 Document Summary")

            st.markdown(
                st.session_state.summary
            )

            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="document_summary.md",
                mime="text/markdown",
                use_container_width=True
            )


# ==================================================
# 11. TAB 3 - QUIZ GENERATOR
# ==================================================

with tab3:

    st.header("📝 AI Quiz Generator")

    st.write(
        "Test your knowledge using questions generated from your PDF."
    )

    if st.session_state.vectorstore is None:

        st.info("Please upload and process a PDF first.")

    else:

        number_of_questions = st.selectbox(
            "Select number of questions",
            [5, 10, 15],
            key="number_of_quiz_questions"
        )

        if st.button(
            "🚀 Generate Quiz",
            use_container_width=True
        ):

            with st.spinner("Generating quiz questions..."):

                try:

                    llm = get_llm()

                    questions = generate_quiz(
                        st.session_state.vectorstore,
                        llm,
                        number_of_questions
                    )

                    if not questions:

                        st.warning(
                            "No quiz questions were generated."
                        )

                    else:

                        st.session_state.quiz_questions = questions
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_answers = {}

                        st.session_state.quiz_version += 1

                        st.success(
                            "Quiz generated successfully!"
                        )

                except Exception as e:

                    st.error(
                        f"Quiz generation failed: {e}"
                    )

        if st.session_state.quiz_questions:

            st.divider()

            st.subheader("📋 Your Quiz")

            if not st.session_state.quiz_submitted:

                with st.form(
                    key=f"quiz_form_{st.session_state.quiz_version}"
                ):

                    answers = {}

                    for i, question in enumerate(
                        st.session_state.quiz_questions
                    ):

                        st.markdown(
                            f"**Q{i + 1}. {question['question']}**"
                        )

                        answers[i] = st.radio(
                            "Choose your answer:",
                            question["options"],
                            key=(
                                f"quiz_answer_"
                                f"{st.session_state.quiz_version}_{i}"
                            ),
                            index=None
                        )

                        st.divider()

                    submitted = st.form_submit_button(
                        "Submit Quiz",
                        use_container_width=True
                    )

                    if submitted:

                        unanswered = [
                            i for i, answer in answers.items()
                            if answer is None
                        ]

                        if unanswered:

                            st.warning(
                                "Please answer all questions before submitting."
                            )

                        else:

                            score = sum(
                                1
                                for i, question in enumerate(
                                    st.session_state.quiz_questions
                                )
                                if answers[i] == question["answer"]
                            )

                            st.session_state.quiz_score = score
                            st.session_state.quiz_answers = answers
                            st.session_state.quiz_submitted = True

                            st.rerun()

            else:

                total = len(
                    st.session_state.quiz_questions
                )

                score = st.session_state.quiz_score

                percentage = (
                    score / total * 100
                    if total else 0
                )

                st.subheader("🏆 Quiz Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Score",
                    f"{score}/{total}"
                )

                col2.metric(
                    "Percentage",
                    f"{percentage:.0f}%"
                )

                result = (
                    "Excellent"
                    if percentage >= 80
                    else "Keep Practicing"
                )

                col3.metric(
                    "Result",
                    result
                )

                st.progress(percentage / 100)

                st.divider()

                st.subheader("Answer Review")

                for i, question in enumerate(
                    st.session_state.quiz_questions
                ):

                    selected = st.session_state.quiz_answers[i]
                    correct = question["answer"]

                    st.markdown(
                        f"**Q{i + 1}. {question['question']}**"
                    )

                    st.write(
                        f"Your answer: {selected}"
                    )

                    st.write(
                        f"Correct answer: {correct}"
                    )

                    if selected == correct:

                        st.success("Correct answer!")

                    else:

                        st.error("Incorrect answer.")

                    st.info(
                        question.get(
                            "explanation",
                            "No explanation provided."
                        )
                    )

                    st.divider()

                if st.button(
                    "🔄 Generate Another Quiz",
                    use_container_width=True
                ):

                    st.session_state.quiz_questions = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_score = 0

                    st.session_state.quiz_version += 1

                    st.rerun()


# ==================================================
# 12. TAB 4 - KEY POINTS EXTRACTOR
# ==================================================

with tab4:

    st.header("🔍 AI Key Points Extractor")

    st.write(
        "Extract important facts, concepts, skills, dates, "
        "projects and findings from your PDF."
    )

    if st.session_state.vectorstore is None:

        st.info("Please upload and process a PDF first.")

    else:

        st.success(
            f"Document loaded: {st.session_state.pdf_name}"
        )

        if st.button(
            "🔎 Extract Key Points",
            use_container_width=True
        ):

            with st.spinner(
                "AI is analyzing your document..."
            ):

                try:

                    llm = get_llm()

                    key_points = extract_key_points(
                        st.session_state.vectorstore,
                        llm
                    )

                    st.session_state.key_points = key_points

                except Exception as e:

                    st.error(
                        f"Key points extraction failed: {e}"
                    )

        if st.session_state.key_points:

            st.divider()

            st.subheader("📌 Extracted Key Points")

            st.markdown(
                st.session_state.key_points
            )

            st.download_button(
                "⬇️ Download Key Points",
                data=st.session_state.key_points,
                file_name="document_key_points.md",
                mime="text/markdown",
                use_container_width=True
            )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.caption(
    "Agentic AI Research Assistant | "
    "Python | Streamlit | LangChain | Groq | FAISS"
)
