from langchain_openai import ChatOpenAI

# Initialize the model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Call the model
response = llm.invoke("What is LangChain?")
print(response)
