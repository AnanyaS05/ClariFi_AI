import streamlit as st
from agent_system import ReActAgent
from language_model import hf_llm, tokenizer
from knowledge_base import TOOLS
from transformers import TextStreamer
import sys
import io
import contextlib

# Custom Streamer for Streamlit
class StreamlitStreamer(TextStreamer):
    def __init__(self, tokenizer, container):
        super().__init__(tokenizer, skip_prompt=True)
        self.container = container
        self.text = ""

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.text += text
        # Update the container with the accumulated text
        # We use a code block or just markdown to show the thought process
        self.container.markdown(f"**Agent Thinking:**\n\n{self.text}")

def main():
    st.set_page_config(page_title="ClariFi AI", page_icon="🔍", layout="wide")
    
    # Sidebar for context
    with st.sidebar:
        st.title("ClariFi AI")
        st.markdown("---")
        st.markdown("**About**")
        st.info(
            "ClariFi AI is an intelligent financial analysis agent powered by ReAct prompting and local LLMs."
        )
        st.markdown("**Capabilities**")
        st.markdown("- 📄 Document Search")
        st.markdown("- 🧠 Reasoning")
        st.markdown("- 📊 Financial QA")
        st.markdown("---")
        st.caption("Powered by Qwen-2.5 & TF-IDF")

    # Main Header
    st.title("ClariFi AI")
    st.subheader("Your Intelligent Financial Analyst")
    
    st.markdown("""
    <style>
    .stTextInput > label {
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("Ask questions about the financial documents in the knowledge base (e.g., Nestle 2024 Report, XYZ Inc.).")

    # Initialize session state for history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Input area
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Enter your financial question:", placeholder="e.g., What were Nestle's total sales in 2024?")
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        analyze_btn = st.button("Analyze Query", type="primary")

    if analyze_btn:
        if not query:
            st.warning("Please enter a question.")
            return

        st.markdown("---")
        
        # Container for streaming thoughts inside an expander
        with st.expander("🧠 Agent Thinking Process", expanded=True):
            thought_container = st.empty()
            thought_container.markdown("*Initializing agent...*")
        
        # Wrapper for LLM to inject the streamer
        streamer = StreamlitStreamer(tokenizer, thought_container)
        
        def streaming_llm(prompt):
            return hf_llm(prompt, streamer=streamer)

        # Initialize Agent
        agent = ReActAgent(llm=streaming_llm, tools=TOOLS)

        with st.spinner("Analyzing financial documents..."):
            try:
                # Run the agent
                result = agent.run(query)
                
                st.markdown("### 🎯 Final Answer")
                if result.get("final_answer"):
                    st.success(result.get("final_answer"))
                else:
                    st.warning("No final answer found.")
                
                st.markdown("---")
                with st.expander("📄 View Full Execution Trace"):
                    st.json(result["steps"])
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
