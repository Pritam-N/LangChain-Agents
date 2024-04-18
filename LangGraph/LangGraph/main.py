from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from agent import (decide_to_generate, 
                  generate, 
                  grade_documents, 
                  grade_generation_v_documents_and_question, 
                  retrieve, 
                  route_question, 
                  transform_query, 
                  web_search)
from models import GraphState

from pprint import pprint
import os

load_dotenv()
os.environ['TAVILY_API_KEY']='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'


def workflow_creater():
    workflow = StateGraph(GraphState)

    workflow.add_node("web_search", web_search) # web search
    workflow.add_node("retrieve", retrieve) # retrieve
    workflow.add_node("grade_documents", grade_documents) # grade documents
    workflow.add_node("generate", generate) # generatae
    workflow.add_node("transform_query", transform_query) # transform_query

    workflow.set_conditional_entry_point(
        route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("retrieve", "grade_documents")   

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    ) 

    workflow.add_edge("transform_query", "retrieve")

    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
        },
    )

    return workflow


def main(question="What are the types of agent memory?"):
    workflow = workflow_creater()

    app = workflow.compile()

    inputs = {"question": question}

    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    # Final generation
    pprint(value ["generation"])

    return value["generation"]

if __name__ == '__main__':
    main()