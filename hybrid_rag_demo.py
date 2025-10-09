from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# 1️⃣ Create sample documents
docs = [
    Document(page_content="Hybrid RAG combines semantic and keyword search for better accuracy."),
    Document(page_content="Vector search retrieves results based on semantic similarity."),
    Document(page_content="BM25 is a sparse retriever using keyword-based matching."),
    Document(page_content="RAG pipelines integrate retrieval and generation steps."),
]

# 2️⃣ Split (optional for large texts)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# 3️⃣ Create two retrievers — Dense + Sparse

# Dense (Vector) Retriever
embedding = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(split_docs, embedding)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Sparse (Keyword/BM25) Retriever
bm25_retriever = BM25Retriever.from_documents(split_docs)

# 4️⃣ Combine them using EnsembleRetriever
# You can assign weights (0.7 dense + 0.3 sparse)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# 5️⃣ Query the Hybrid Retriever
query = "How does hybrid RAG work?"
results = hybrid_retriever.get_relevant_documents(query)

print("🔍 Retrieved Documents:")
for i, d in enumerate(results, start=1):
    print(f"{i}. {d.page_content}")
