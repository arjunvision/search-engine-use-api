import streamlit as st
import openai
import wikipedia
import arxiv

st.set_page_config(page_title="ChatGPT Search Agent", page_icon="🔎")
st.title("🔎 ChatGPT — Chat with Search")
st.caption("Powered by GPT-4o-mini + Wikipedia + Arxiv")

st.sidebar.title("⚙️ Settings")
api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
if not api_key:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if not api_key:
    st.sidebar.warning("Please enter your OpenAI API key.")
    st.stop()

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=4, auto_suggest=True)
    except Exception as e:
        return f"Wikipedia: no result ({e})"

def search_arxiv(query):
    try:
        client = arxiv.Client()
        results = list(client.results(arxiv.Search(query=query, max_results=2)))
        if not results:
            return "No arxiv papers found."
        return "\n\n".join(
            f"Title: {r.title}\nSummary: {r.summary[:300]}..." for r in results
        )
    except Exception as e:
        return f"Arxiv: no result ({e})"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything — I'll search Wikipedia and Arxiv for you!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("e.g. What is deep learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            wiki_result  = search_wikipedia(prompt)
            arxiv_result = search_arxiv(prompt)

            context = f"""Answer the user's question using these search results.

### Wikipedia:
{wiki_result}

### Arxiv Papers:
{arxiv_result}

### Question:
{prompt}

Give a clear and helpful answer."""

            client = openai.OpenAI(api_key=api_key)
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions using search results."},
                    {"role": "user",   "content": context}
                ],
                temperature=0,
                stream=True
            )

            answer = ""
            placeholder = st.empty()
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                answer += delta
                placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
