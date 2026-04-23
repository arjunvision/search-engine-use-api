import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# ── Tools Setup ───────────────────────────────────────────────
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun()

# ── ReAct Prompt ──────────────────────────────────────────────
REACT_TEMPLATE = """Answer the following questions as best you can. \
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(REACT_TEMPLATE)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="ChatGPT Search Agent", page_icon="🔎")
st.title("🔎 ChatGPT — Chat with Search")
st.caption("Powered by GPT-4o-mini + LangChain + DuckDuckGo, Arxiv, Wikipedia")

# ── Sidebar: API Key ──────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key to start.")
    st.stop()

# ── Chat History ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I can search the web for you. Ask me anything!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input(placeholder="e.g. What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-4o-mini",
        temperature=0,
        streaming=True
    )

    tools = [search, arxiv, wiki]

    agent = create_react_agent(llm, tools, react_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=5           # prevents infinite loops
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False
        )
        try:
            response = agent_executor.invoke(
                {"input": prompt},
                config={"callbacks": [st_cb]}
            )
            answer = response["output"]
        except Exception as e:
            answer = f"Sorry, I ran into an error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)
