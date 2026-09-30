from pathlib import Path
import streamlit as st
from rag import load_index, get_llm, retrieve, answer, rewrite, REWRITE

st.set_page_config(page_title="Chess Opening Assistant", page_icon="♟️")
st.title("♟️ Chess Opening Assistant")
st.caption("Answers come only from the Lichess openings database (3,815 openings).")

if not Path("index").exists():
    st.error("No index found. Run `python build_index.py` first.")
    st.stop()


@st.cache_resource  # load the index and model once, not on every question
def setup():
    return load_index(), get_llm()


index, llm = setup()

question = st.text_input("Ask about an opening", placeholder="e.g. What are the moves of the Ruy Lopez?")

if question:
    with st.spinner("Thinking..."):
        search_query = rewrite(question, llm) if REWRITE else None
        results = retrieve(index, question, extra_query=search_query)
        reply = answer(question, results, llm)
    st.markdown(reply)

    with st.expander("Openings the answer was based on"):
        if search_query:
            st.caption(f"Also searched for: {search_query}")
        for doc, score in results:
            st.markdown(
                f"**{doc.metadata['eco']} · {doc.metadata['name']}** (score {score:.3f})  \n"
                f"`{doc.metadata['pgn']}`"
            )