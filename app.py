"""Run with: python -m streamlit run app.py"""
import streamlit as st

def main():
    st.set_page_config(page_title="Insurance Document Assistant", page_icon="📚", layout="wide")
    try:
        from frontend.streamlit_ui import run_app
        run_app()
    except Exception as exc:
        from utils.logger import get_logger
        get_logger(__name__).error("Startup failed (%s)", type(exc).__name__)
        st.error("Application startup failed. Run python -m scripts.doctor from the project folder for diagnostics.")

if __name__ == "__main__":
    main()
