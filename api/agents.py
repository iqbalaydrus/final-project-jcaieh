import os
from typing import Optional

import sqlite3

from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

# New imports for the agents
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

import schema

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

db: Optional[sqlite3.Connection] = None
qdrant: Optional[QdrantVectorStore] = None

# Create and return SQL agent
def get_sql_agent(llm, db_path='database.db'):
    """Creates a LangChain SQL Agent."""
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent_executor

# Chat function that acts as the MAIN AGENT ---
def chat(req: schema.ChatRequest, cv_file_contents: Optional[str]) -> str:
    """
    This is the main agent that routes questions to the appropriate specialist agent.
    """
    user_question = req.message.content
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

    # 1. Router: Decide which agent to use
    router_prompt_template = """
    You are an expert query router. Your job is to determine whether a user's question should be answered by a RAG agent or a SQL agent.
    - The RAG agent handles semantic questions about job descriptions, responsibilities, skills, and qualifications.
    - The SQL agent handles factual questions that can be answered from a database table with columns like 'work_type', 'salary', 'location', 'company_name', and 'job_title'.

    Based on the user's question below, respond with "RAG" or "SQL".

    User Question: "{question}"
    """
    router_prompt = PromptTemplate.from_template(router_prompt_template)
    router_chain = LLMChain(llm=llm, prompt=router_prompt)

    # Assumed the RAG agent isn't ready and just handle the SQL case.
    
    # Placeholder for the RAG agent's response
    rag_response = "I am the RAG agent. I can answer questions about job descriptions and qualifications. This feature is being integrated."

    try:
        # Default to SQL agent if the routing fails
        route = router_chain.run(user_question)
        print(f"[{req.session_id}] Router decided: {route}")

        if "sql" in route.lower():
            print(f"[{req.session_id}] --- Activating SQL Agent ---")
            sql_agent = get_sql_agent(llm)
            # Pass the user question directly to the SQL agent
            agent_response = sql_agent.run(user_question)
            return agent_response
        else:
            print(f"[{req.session_id}] --- Activating RAG Agent (Placeholder) ---")
            # When RAG agent is ready, will call it here.
            return rag_response

    except Exception as e:
        print(f"[{req.session_id}] An error occurred in the main agent: {e}")
        return "Sorry, I encountered an error. Please try rephrasing your question."
