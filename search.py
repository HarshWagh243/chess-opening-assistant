import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index = FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)

query = " ".join(sys.argv[1:]) or "grunfeld moves"
print(f"Query: {query}\n")

# Lower score = closer match (FAISS measures distance)
for doc, score in index.similarity_search_with_score(query, k=4):
    print(f"{score:.3f}  {doc.metadata['eco']}  {doc.metadata['name']}")