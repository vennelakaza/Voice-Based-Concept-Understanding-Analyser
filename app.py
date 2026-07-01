import streamlit as st

st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Voice-Based Concept Understanding Analyser")

st.write("Welcome to the VBCUA Project!")

st.markdown("---")

st.header("Project Initialized Successfully")

st.write("""
This application evaluates conceptual understanding using voice.

Features:
- Speech-to-Text (Whisper)
- Semantic Similarity
- Audio Analysis
- PDF Report Generation
""")
