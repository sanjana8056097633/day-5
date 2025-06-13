import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

# ✅ Directly assign your Google Gemini API key here (for demo only)
GOOGLE_API_KEY = "AIzaSyBHzx-k_9qtFuxcJAX1GG6FbH3KdR7vDzQ"

# Check if API key is provided
if not GOOGLE_API_KEY:
    st.error("❌ Google Gemini API key is missing.")
    st.stop()

# Initialize Gemini LLM via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translation assistant that translates English to French."),
    ("user", "Translate the following sentence to French: {english_sentence}")
])

# Create a simple chain using prompt | llm
translation_chain: Runnable = prompt | llm

# Streamlit UI
st.set_page_config(page_title="English to French Translator", page_icon="🌍")
st.title("🌍 English to French Translator")

# Text input
english_input = st.text_input("Enter an English sentence:")

# Translate button
if st.button("Translate"):
    if not english_input.strip():
        st.warning("⚠️ Please enter a valid English sentence.")
    else:
        try:
            # Run the chain
            result = translation_chain.invoke({"english_sentence": english_input})
            # Extract and display translation
            translation = result.content if hasattr(result, 'content') else str(result)
            st.success("✅ Translation Successful!")
            st.text_area("French Translation:", value=translation, height=100)
        except Exception as e:
            st.error(f"❌ Error occurred during translation:\n{e}")
