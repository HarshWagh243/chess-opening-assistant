import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()  # reads OPENAI_API_KEY from .env

# 1. Load all five TSV files into one table
files = sorted(Path("data").glob("*.tsv"))
df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)

# 2. One opening = one document (no chunking needed)
docs = []
for row in df.itertuples():
    text = f"{row.name} (ECO {row.eco}). Moves: {row.pgn}"
    docs.append(Document(page_content=text, metadata={"eco": row.eco, "name": row.name, "pgn": row.pgn}))

print(f"Built {len(docs)} documents")
print("Example:", docs[0].page_content)

# 3. Embed every document and store the vectors in a FAISS index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index = FAISS.from_documents(docs, embeddings)

# 4. Save to disk so we only pay for embeddings once
index.save_local("index")
print("Saved index to the index/ folder")