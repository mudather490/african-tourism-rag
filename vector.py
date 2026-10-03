# vector.py
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# 1. Load the CSV
loader = CSVLoader(
    file_path="Tourism in Africa.csv",
    encoding="utf-8",
    csv_args={"delimiter": ",", "quotechar": '"'},
)
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)

# 3. Embeddings
embeddings = OllamaEmbeddings(model="embeddinggemma")

# 4. Build or load the vector store
if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="langchain",
    )
    print("Loaded existing Chroma DB.")
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name="langchain",
    )
    print("Created new Chroma DB from CSV.")

# 5. Retriever (exposed for main.py)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# Quick test when running this file directly
if __name__ == "__main__":
    results = retriever.invoke("What is the best time to visit Rwanda?")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:300])
        print()