import streamlit as st
from pypdf import PdfReader
import os

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import create_agent

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# AI RESUME ANALYZER
# --------------------------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("🤖 AI Resume Analyzer Agent")
st.write("Upload your resume and let AI analyze your skills, education, projects and experience.")


# --------------------------------------------------
# Gemini API
# --------------------------------------------------

GOOGLE_API_KEY = os.environ["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# --------------------------------------------------
# Resume Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your Resume PDF",
    type=["pdf"],
    key="resume_uploader"
)


if uploaded_file is not None:

    # --------------------------------------------------
    # Extract PDF Text
    # --------------------------------------------------

    reader = PdfReader(uploaded_file)

    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text + "\n"

    if not resume_text.strip():
        st.error("Could not extract text from this PDF.")
        st.stop()


    # --------------------------------------------------
    # Create Document
    # --------------------------------------------------

    documents = [
        Document(page_content=resume_text)
    ]


    # --------------------------------------------------
    # Split Resume into Chunks
    # --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)


    # --------------------------------------------------
    # Create FAISS Vector Store
    # --------------------------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )


    # --------------------------------------------------
    # Resume Retrieval Tool
    # --------------------------------------------------

    @tool(response_format="content_and_artifact")
    def retrieve_resume_context(query: str):
        """Retrieve relevant information from the uploaded resume."""

        retrieved_docs = vector_store.similarity_search(
            query,
            k=3
        )

        serialized = "\n\n".join(
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )

        return serialized, retrieved_docs


    # --------------------------------------------------
    # Create Resume Analyzer Agent
    # --------------------------------------------------

    tools = [retrieve_resume_context]

    prompt = (
        "You are an AI Resume Analyzer Agent. "
        "Use the resume retrieval tool to analyze the uploaded resume. "
        "Identify the candidate's skills, education, projects, "
        "experience, certifications and achievements. "
        "Give clear and structured answers. "
        "Do not invent information that is not present in the resume."
    )

    resume_agent = create_agent(
        llm,
        tools,
        system_prompt=prompt
    )


    # --------------------------------------------------
    # Analyze Resume
    # --------------------------------------------------

    if st.button("🔍 Analyze Resume"):

        with st.spinner("Analyzing resume..."):

            query = """
            Analyze this resume and provide:

            1. Technical Skills
            2. Education
            3. Projects
            4. Work Experience
            5. Certifications
            6. Achievements
            7. Overall Resume Summary

            Use only information available in the uploaded resume.
            """

            try:

                result = resume_agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    }
                )

                message = result["messages"][-1]

                if isinstance(message.content, list):

                    answer = ""

                    for content in message.content:

                        if content.get("type") == "text":
                            answer += content["text"]

                else:
                    answer = message.content

                st.success("Resume analysis completed!")

                st.markdown("## 📊 Resume Analysis")

                st.write(answer)

            except Exception as e:

                st.error("Error while analyzing the resume.")
                st.exception(e)
