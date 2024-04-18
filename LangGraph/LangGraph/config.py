
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_openai import AzureChatOpenAI

temperature=0
max_tokens=1000
top_p=0.9
frequency_penalty=0
presence_penalty=0
stop=None


embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
llm = AzureChatOpenAI(
    openai_api_key='7425866287b14892ac27f27785acd412',
    openai_api_version="2024-02-15-preview",
    azure_deployment="chat35turbo",
    azure_endpoint="https://novaplayground.openai.azure.com/",
    temperature=0
)
rag_prompt = hub.pull("rlm/rag-prompt")