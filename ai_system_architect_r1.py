import time
import streamlit as st
from openai import OpenAI
import json

# Model IDs via OpenRouter
DEEPSEEK_MODEL = "deepseek/deepseek-r1-0528:free"
MISTRAL_MODEL = "mistralai/devstral-small:free"


class ModelChain:
    def __init__(self, deepseek_api_key: str, mistral_api_key: str):
        self.deepseek = OpenAI(api_key=deepseek_api_key,
                               base_url="https://openrouter.ai/api/v1")
        self.mistral = OpenAI(api_key=mistral_api_key,
                              base_url="https://openrouter.ai/api/v1")

    def get_deepseek_reasoning(self, prompt: str) -> str:
        try:
            response = self.deepseek.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a software architecture expert. Reply ONLY with JSON per schema."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ DeepSeek Error: {str(e)}"

    def get_mistral_response(self, user_input: str, deepseek_json: str) -> str:
        try:
            response = self.mistral.chat.completions.create(
                model=MISTRAL_MODEL,
                messages=[
                    {"role": "system", "content": "You are a software architect. Explain architecture based on JSON from DeepSeek."},
                    {"role": "user", "content": f"User Query: {user_input}\nDeepSeek Output:\n{deepseek_json}"}
                ],
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"❌ Mistral Error: {str(e)}"


def main():
    st.set_page_config(page_title="System Architect Advisor", layout="wide")
    st.title("🤖 AI System Architect Advisor ")

    # Sidebar: API Keys
    deep_key = st.sidebar.text_input("🔑 DeepSeek API Key", type="password")
    mistral_key = st.sidebar.text_input("🔐 Mistral API Key", type="password")

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Prompt Guide
    st.info("""
    💡 **Prompt Tips**  
    Structure your query like this for best results:

    - **What are you building?**  
    - **Key requirements** (functional + non-functional)  
    - **Constraints** (e.g., team size, infra, budget, timeline)  
    - **Scale** (expected usage, data volume)  
    - **Security/Compliance** (e.g., HIPAA, SOC2)

    **Example Prompt:**  
    I need to build a hospital management system that:
    - Handles patients, appointments, billing, and EHR  
    - Must be HIPAA compliant  
    - Should scale across 100 rural clinics  
    - Requires offline access & syncing  
    - Budget: $80,000 | Team: 5 | Timeline: 6 months
    """)

    # Chat History Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Prompt
    if prompt := st.chat_input("Describe your architecture needs…"):
        if not deep_key or not mistral_key:
            st.error("⚠️ Please provide both API keys in the sidebar.")
            return

        chain = ModelChain(deep_key, mistral_key)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("🛠️ Generating DeepSeek architecture analysis..."):
                deep_json = chain.get_deepseek_reasoning(prompt)

            with st.expander("📦 DeepSeek JSON Output", expanded=False):
                st.code(deep_json, language="json")

            with st.spinner("💡 Generating Mistral explanation..."):
                explanation = chain.get_mistral_response(prompt, deep_json)

            st.markdown(explanation)
            st.session_state.messages.append(
                {"role": "assistant", "content": explanation})

            # Optional Download
            st.download_button("📥 Download Explanation", explanation,
                               file_name="architecture_explanation.md")


if __name__ == "__main__":
    main()
