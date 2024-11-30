from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# 定义 Prompt 模板
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

# 初始化模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 创建链
chain = prompt | model

# 调用链
result = chain.invoke({"topic": "bears"})
print(result)
