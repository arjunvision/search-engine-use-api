import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# ── Tools Setup ──────────────────────────────────────────────
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="ChatGPT Search Agent", page_icon="🔎")
st.title("🔎 ChatGPT — Chat with Search")
st.caption("Powered by OpenAI GPT-4 + LangChain + DuckDuckGo, Arxiv, Wikipedia")

# ── Sidebar: API Key ──────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# Try secrets first (Streamlit Cloud), then let user input
api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""

if not api_key:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key to start.")
    st.stop()

# ── Chat History ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a ChatGPT-powered bot that can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input(placeholder="Ask me anything... e.g. What is RAG in AI?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ── LLM + Agent ───────────────────────────────────────────
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4o-mini",   # cheaper & fast; change to "gpt-4o" for best quality
        temperature=0,
        streaming=True
    )

    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
