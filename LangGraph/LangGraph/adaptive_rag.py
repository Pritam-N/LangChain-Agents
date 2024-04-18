from typing import Literal
import re

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain.output_parsers import PydanticOutputParser

from dotenv import load_dotenv
import os

from models import GradeAnswer, GradeDocuments, GradeHallucinations, RouteQuery
from config import llm, embedding, rag_prompt

def preprocess_json_string(json_str: str) -> str:
    # This is a simplistic approach and might not cover all edge cases.
    # It adds double quotes around any sequence of characters that is immediately
    # followed by a colon and spaces, assuming it's a key.
    corrected_str = re.sub(r'(?<!")(\b\w+\b)(?!":)', r'"\1"', json_str)
    return corrected_str

def preprocess_json_string_with_newline(json_str: str) -> str:
    # This is a simplistic approach and might not cover all edge cases.
    # It adds double quotes around any sequence of characters that is immediately
    # followed by a colon and spaces, assuming it's a key.
    corrected_str = re.sub(r'(?<!")(\b\w+\b)(?!":)', r'"\1"', json_str)
    corrected_str = corrected_str.replace('\n', '')
    corrected_str = corrected_str.replace("\\'", "").replace('\'"', '"').replace('"\'', '"')

    return corrected_str

def vector_retriever():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    doc_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size= 500,
        chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(doc_list)
    vector_store = Chroma.from_documents(
        embedding=embedding,
        documents=doc_splits,
        collection_name='rag-chroma'
    )
    retriever = vector_store.as_retriever()
    return retriever
    
def question_router():
    # structured_llm_router = llm.with_structured_output(RouteQuery, include_raw=True)

    # system="""
    # You are an expert in routing user questions to a vector store or web search. 
    # The vector store contains documents related to agents, prompt engineering and adversarial attacks.
    # Use the vector-store for questions on these topics. Other use web-search.
    # Do not return a function or other content. Only return RouteQuery datasource as 'vector_store' or 'websearch' in curly braces
    # """

    # route_template = ChatPromptTemplate.from_messages(
    #     [('system', system),
    #     ('human', '{question}')]
    # )
    route_parser = PydanticOutputParser(pydantic_object=RouteQuery)

    system="""
    You are an expert in routing user questions to a vector store or web search. 
    The vector store contains documents related to agents, prompt engineering and adversarial attacks.
    Use the vector-store for questions on these topics. Other use web-search.
    """

    prompt = PromptTemplate(
        template= system + '\n {format_instructions} \n {query} \n',
        input_variables=['query'],
        partial_variables={'format_instructions': route_parser.get_format_instructions()}
    )

    chain = prompt | llm | route_parser

    return chain

def decider_template(user_query="What are the types of agent memory?"):

    output = question_router().invoke({"query": user_query})

    # parsed_path = RouteQuery.parse_raw(
    #     preprocess_json_string(
    #         output["raw"].additional_kwargs["tool_calls"][0]['function']['arguments']))
    
    return output.datasource

def retrieval_grader():
    # structured_llm_grader = llm.with_structured_output(GradeDocuments, include_raw=True)

    # # Prompt 
    # system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    #     If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    #     It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    #     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    #     Donot return a function or other content. Only return binary score in curly braces"""
    # grade_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system),
    #         ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    #     ]
    # )

    # grader = grade_prompt | structured_llm_grader

    retrieval_parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    prompt = PromptTemplate(
        template= system + '\n {format_instructions} \n Retrieved document: {document} \n\n User question: {question} \n',
        input_variables=['question', 'document'],
        partial_variables={'format_instructions': retrieval_parser.get_format_instructions()}
    )

    retriever = prompt | llm | retrieval_parser

    return retriever

def document_grader_template(question = "agent memory", docs=''):
    
    # docs = vector_retriever().get_relevant_documents(question)
    # doc_txt = docs[1].page_content

    output = retrieval_grader().invoke({"question": question, "document": docs})

    # parsed_datasource = GradeDocuments.parse_raw(
    #     preprocess_json_string_with_newline(
    #         output["raw"].additional_kwargs["tool_calls"][0]['function']['arguments']))

    return output.binary_score

def rag_chain():
    # Prompt
    prompt = rag_prompt

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    chain = prompt | llm | StrOutputParser()

    return chain

def generate_response_template(question="agent memory"):
    
    docs = vector_retriever().get_relevant_documents(question)

    # Run
    generation = rag_chain().invoke({"context": docs, "question": question})

    return generation

def hallucination_grader():
    # structured_llm_grader = llm.with_structured_output(GradeHallucinations, include_raw=True)

    # # Prompt 
    # system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    #     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    # hallucination_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system),
    #         ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    #     ]
    # )

    # grader = hallucination_prompt | structured_llm_grader
    hallucination_parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

    # Prompt 
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

    prompt = PromptTemplate(
        template= system + '\n {format_instructions} \n Set of facts: {documents} \n LLM generation: {generation} \n',
        input_variables=['generation', 'documents'],
        partial_variables={'format_instructions': hallucination_parser.get_format_instructions()}
    )

    chain = prompt | llm | hallucination_parser
    return chain

def hallucination_grader_template(documents, generation):
    # docs = vector_retriever().get_relevant_documents(question)
    
    output = hallucination_grader().invoke({"documents": documents, "generation": generation})

    # parsed_datasource = GradeHallucinations.parse_raw(
    #     preprocess_json_string_with_newline(
    #         output["raw"].additional_kwargs["tool_calls"][0]['function']['arguments']))

    return output.binary_score

def answer_grader():
    # structured_llm_grader = llm.with_structured_output(GradeAnswer, include_raw=True)

    # # Prompt 
    # system = """You are a grader assessing whether an answer addresses / resolves a question \n 
    #     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
    #     Donot return a function or other content. Only return binary score in curly braces."""
    # answer_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system),
    #         ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    #     ]
    # )

    # grader = answer_prompt | structured_llm_grader

    answer_parser = PydanticOutputParser(pydantic_object=GradeAnswer)

    system = """You are a grader assessing whether an answer addresses / resolves a question \n 
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    
    prompt = PromptTemplate(
        template= system + '\n {format_instructions} \n User question: {question} \n LLM generation: {generation} \n',
        input_variables=['generation', 'question'],
        partial_variables={'format_instructions': answer_parser.get_format_instructions()}
    )

    chain = prompt | llm | answer_parser
    return chain

def answer_grader_template(question = "agent memory", generation=""):
    
    output = answer_grader().invoke({"question": question,"generation": generation})

    # parsed_datasource = GradeAnswer.parse_raw(
    #     preprocess_json_string_with_newline(
    #         output["raw"].additional_kwargs["tool_calls"][0]['function']['arguments']))

    return output.binary_score

def question_rewriter():
    # Prompt 
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ]
    )

    rewriter = re_write_prompt | llm | StrOutputParser()

    return rewriter

def rewrite_question_template(question):
    output = question_rewriter().invoke({"question": question})
    return output

def web_search_tool():
    return TavilySearchResults(k=5)
