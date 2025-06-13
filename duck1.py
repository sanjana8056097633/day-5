import streamlit as st
import logging
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI  # <--- fixed import

# --- Suppress verbose logs ---
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# --- Hardcoded Gemini API key ---
GEMINI_API_KEY = "AIzaSyCtD7pFRnyEX-0BxEvqI7QLpHl9fz_VWYw"

# --- Initialize Gemini chat model ---
def get_gemini_model():
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return None

# --- Initialize DuckDuckGo Search Tool ---
def get_search_tool():
    try:
        return DuckDuckGoSearchResults()
    except Exception as e:
        st.error(f"Error initializing search tool: {e}")
        return None

# --- Initialize agent with tools ---
def init_agent(model, tools):
    try:
        agent = initialize_agent(
            tools,
            model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            max_iterations=3,
        )
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="🦜 Gemini Real-Time Q&A", page_icon="🧠", layout="centered")

st.title("🦜 Gemini Real-Time Q&A with DuckDuckGo Search")
st.write("Ask any question about current events or facts, and get real-time answers! 🔍🌐")

user_question = st.text_input("Enter your question here:", placeholder="e.g., What's the latest news on AI?")

if st.button("Ask Gemini"):
    if not user_question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking... 🧠"):
            model = get_gemini_model()
            if model:
                search_tool = get_search_tool()
                if search_tool:
                    agent = init_agent(model, [search_tool])
                    if agent:
                        try:
                            answer = agent.run(user_question)
                            st.markdown("### Answer:")
                            st.write(answer)
                        except Exception as err:
                            st.error(f"Oops! Something went wrong while generating the answer: {err}")
                else:
                    st.error("Search tool initialization failed.")
            else:
                st.error("Gemini model initialization failed.")