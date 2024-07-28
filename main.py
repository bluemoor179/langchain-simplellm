# Use Language Model
from langchain_community.chat_models import ChatOllama

model = ChatOllama(model='llama3.1')

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

result = model.invoke(messages)

# Output Parser
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

parser.invoke(result)

# Chain
chain = model | parser

chain.invoke(messages)

# Prompt Templates
from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

result = prompt_template.invoke({"language": "italian", "text": "hi"})

result.to_messages()

# Chaining together components with LCEL
chain = prompt_template | model | parser
print(chain.invoke({"language": "italian", "text": "hi"}))