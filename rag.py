import re
import unicodedata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.4-mini"
K = 4          # how many openings the model sees
POOL = 20      # how many candidates each search method returns before fusion
MODE = "keyword"  # "vector", "keyword" or "hybrid"
REWRITE = True
PROMPT = """You are a chess opening assistant. Answer the question using ONLY the openings listed below.

Rules:
- Cite every opening you use as [ECO code, name].
- If the answer is not in the list, reply exactly: "I don't know - that isn't in my openings database."
- Do not use your own chess knowledge, even if you know the answer.

Openings:
{context}

Question: {question}"""

REWRITE_PROMPT = """You turn chess questions into search queries for a database of chess openings.
The database only contains opening names (e.g. "Sicilian Defense: Najdorf Variation"),
ECO codes (e.g. B90) and moves in standard notation (e.g. "e4 c5 Nf3 d6").

Write a short search query that would find the right opening:
- If you can tell which opening is meant, include its standard English name.
- If the question describes moves, write them in standard notation (e.g. "the king moves on move 2" -> "Ke2").
- Return only the query, nothing else.

Question: {question}"""


def load_documents():
    """One opening = one document. Used by build_index.py and by keyword search."""
    files = sorted(Path("data").glob("*.tsv"))
    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    return [
        Document(
            page_content=f"{row.name} (ECO {row.eco}). Moves: {row.pgn}",
            metadata={"eco": row.eco, "name": row.name, "pgn": row.pgn},
        )
        for row in df.itertuples()
    ]


# Common question words that carry no chess meaning. Without this list, "the" in
# "moves of the Ruy Lopez" matched openings like "Zukertort Opening: The Potato".
STOPWORDS = {
    "a", "an", "the", "of", "is", "are", "what", "which", "how", "does", "do", "go",
    "me", "show", "i", "played", "this", "that", "it", "s", "for", "in", "on", "to",
    "called", "moves", "move", "opening", "line", "code", "eco", "refer", "with",
}


def tokenize(text):
    """Split text into search words: 'Grünfeld' -> 'grunfeld', '1.d4' -> 'd4'."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\d+\.", " ", text)  # drop move numbers like "1." or "12."
    return [w for w in re.findall(r"[a-z0-9]+", text) if w not in STOPWORDS]


def load_index():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    docs = load_documents()
    return {
        "faiss": FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True),
        "bm25": BM25Okapi([tokenize(d.page_content) for d in docs]),
        "docs": docs,
    }


def get_llm():
    return ChatOpenAI(model=CHAT_MODEL)


def vector_search(index, question, n):
    return [doc for doc, _ in index["faiss"].similarity_search_with_score(question, k=n)]


def keyword_search(index, question, n):
    scores = index["bm25"].get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [index["docs"][i] for i in top]


def fuse(ranked_lists, c=60):
    """Reciprocal Rank Fusion: a document scores 1/(c + rank) in each list it appears in."""
    scores, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content
            docs[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
    best = sorted(scores, key=scores.get, reverse=True)
    return [(docs[key], scores[key]) for key in best]


def rewrite(question, llm):
    """Ask the LLM to turn a question into names and moves the database actually contains."""
    return llm.invoke(REWRITE_PROMPT.format(question=question)).text.strip()


def retrieve(index, question, k=K, mode=MODE, extra_query=None):
    """Return the k best openings as (document, score) pairs.

    extra_query: an optional rewritten query, searched alongside the original question.
    """
    if mode == "vector" and not extra_query:
        return index["faiss"].similarity_search_with_score(question, k=k)  # score = distance, lower is better

    queries = [question] + ([extra_query] if extra_query else [])
    ranked_lists = []
    for q in queries:
        if mode in ("keyword", "hybrid"):
            ranked_lists.append(keyword_search(index, q, POOL))
        if mode in ("vector", "hybrid"):
            ranked_lists.append(vector_search(index, q, POOL))
    return fuse(ranked_lists)[:k]

def answer(question, results, llm):
    """Send the retrieved openings plus the question to the model."""
    context = "\n".join(
        f"- [{doc.metadata['eco']}, {doc.metadata['name']}] {doc.metadata['pgn']}"
        for doc, _ in results
    )
    prompt = PROMPT.format(context=context, question=question)
    return llm.invoke(prompt).text