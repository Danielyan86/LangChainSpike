from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load PDF document
loader = PyPDFLoader("baichuan2_test.pdf")  # Replace with your PDF path
pages = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, separator="\n"
)
chunks = text_splitter.split_documents(pages)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

# Perform retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 relevant chunks
)

# Example query
query = "what is baichuan2 ？"  # Replace with your question
docs = retriever.invoke(query)

# Print results
print("\n=== 检索结果 ===")
for i, doc in enumerate(docs, 1):
    print(f"\n文档片段 {i}:")
    print("-" * 50)
    print(f"内容: {doc.page_content}")
    print(f"\n页码: {doc.metadata.get('page', 'Unknown')}")
