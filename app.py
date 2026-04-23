import streamlit as st
import openai
import wikipedia
import arxiv
from duckduckgo_search import DDGS

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="ChatGPT Search Agent", page_icon="🔎")
st.title("🔎 ChatGPT — Chat with Search")
st.caption("Powered by GPT-4o-mini + DuckDuckGo + Wikipedia + Arxiv")

# ── Sidebar: API Key ──────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key.")
    st.stop()

# ── Search Helper Functions ───────────────────────────────────
def search_web(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"**{r['title']}**\n{r['body']}" for r in results
        )
    except Exception as e:
        return f"Search error: {str(e)}"

def search_wikipedia(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=3)
        return summary
    except Exception as e:
        return f"Wikipedia error: {str(e)}"

def search_arxiv(query: str) -> str:
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=2)
        results = list(client.results(search))
        if not results:
            return "No arxiv papers found."
        return "\n\n".join(
            f"**{r.title}**\n{r.summary[:300]}..." for r in results
        )
    except Exception as e:
        return f"Arxiv error: {str(e)}"

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

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                # Step 1: Search all three sources
                web_results     = search_web(prompt)
                wiki_results    = search_wikipedia(prompt)
                arxiv_results   = search_arxiv(prompt)

                # Step 2: Build context for GPT
                context = f"""Use the following search results to answer the user's question.

### Web Search Results:
{web_results}

### Wikipedia:
{wiki_results}

### Arxiv Papers:
{arxiv_results}

### User Question:
{prompt}

Answer clearly and concisely based on the above information."""

                # Step 3: Ask GPT-4o-mini
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers questions using search results."},
                        {"role": "user",   "content": context}
                    ],
                    temperature=0,
                    stream=True
                )

                # Step 4: Stream the response
                answer = ""
                placeholder = st.empty()
                for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    answer += delta
                    placeholder.markdown(answer)

            except Exception as e:
                answer = f"Error: {str(e)}"
                st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
