from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

# Join the texts into a single string with newlines
texts = ["LangChain simplifies AI application development.", "It supports retrieval-based generation."]
combined_text = "\n".join(texts)

# Text splitting
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = text_splitter.split_text(combined_text)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)

# Retrieval-augmented generation
retriever = vectorstore.as_retriever()
retrieved_docs = retriever.invoke("What is LangChain?")

# Format and print the results
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nDocument {i}:")
    print("-" * 50)
    print(f"Content: {doc.page_content}")
    if doc.metadata:
        print("\nMetadata:")
        for key, value in doc.metadata.items():
            print(f"{key}: {value}")
