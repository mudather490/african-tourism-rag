# main.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import the retriever we already built in vector.py
from vector import retriever


# 1. Define the model
model = OllamaLLM(model="llama3.2:3b")


# 2. Prompt template
template = """
You are an expert in answering questions about African Tourism in Egypt, Rwanda, and South Africa.

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


# 3. Helper to format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 4. Build the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# 5. Interactive loop
if __name__ == "__main__":
    print("African Tourism RAG — type 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}\n")

