from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.4-mini"
K = 4  # how many openings to retrieve per question

PROMPT = """You are a chess opening assistant. Answer the question using ONLY the openings listed below.

Rules:
- Cite every opening you use as [ECO code, name].
- If the answer is not in the list, reply exactly: "I don't know - that isn't in my openings database."
- Do not use your own chess knowledge, even if you know the answer.

Openings:
{context}

Question: {question}"""


def load_index():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)


def get_llm():
    return ChatOpenAI(model=CHAT_MODEL)


def retrieve(index, question, k=K):
    """Return the k closest openings as (document, distance) pairs."""
    return index.similarity_search_with_score(question, k=k)


def answer(question, results, llm):
    """Send the retrieved openings plus the question to the model."""
    context = "\n".join(
        f"- [{doc.metadata['eco']}, {doc.metadata['name']}] {doc.metadata['pgn']}"
        for doc, _ in results
    )
    prompt = PROMPT.format(context=context, question=question)
    return llm.invoke(prompt).text