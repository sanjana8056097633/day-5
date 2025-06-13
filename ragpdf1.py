import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from tempfile import NamedTemporaryFile

# 🔐 Hardcoded API Key for demo/testing (replace with your own)
GOOGLE_API_KEY = "AIzaSyBHzx-k_9qtFuxcJAX1GG6FbH3KdR7vDzQ"

# ✅ Initialize Gemini 2.0 Flash via LangChain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# 🌐 Streamlit UI
st.set_page_config(page_title="PDF QA with Gemini (No VectorDB)", page_icon="📄")
st.title("📄 Ask Questions from a PDF (No Embeddings Needed)")

# 📂 PDF Upload
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    # 🔍 Load and extract text from PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # 📄 Join all pages into one large string (or chunk if needed)
    all_text = "\n\n".join([page.page_content for page in pages])

    # ✏️ Ask a question
    user_question = st.text_input("Ask a question from the PDF:")

    if user_question:
        try:
            # 🧠 Define the prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant answering questions using the given context."),
                ("user", "Context:\n{document}\n\nQuestion:\n{question}")
            ])

            # 🔗 Chain
            chain = prompt | llm

            # ✨ Invoke the chain
            result = chain.invoke({"document": all_text[:8000], "question": user_question})  # limit tokens if needed

            # ✅ Show result
            st.markdown("### ✅ Answer:")
            st.write(result.content if hasattr(result, "content") else str(result))

            # 📑 Optionally display document preview
            with st.expander("📄 Show extracted PDF content"):
                st.text_area("PDF Content", value=all_text[:3000], height=300)

        except Exception as e:
            st.error(f"❌ Error: {e}")
