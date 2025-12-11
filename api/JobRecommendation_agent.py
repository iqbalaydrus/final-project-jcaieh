import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent

# ===== SECRET KEY =====
QDRANT_URL = os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ===== llm & embedding setup =====
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    api_key=OPENAI_API_KEY)

# ===== QDRANT COLLECTION =====
collection_name = "indonesian_job_v2"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ===== JOB RECOMMENDATION TOOL =====
@tool
def job_recommendation(user_profile: str) -> str:
    """
    Recommend suitable job positions based on user's educational background and skills.
    """
    results = qdrant.similarity_search(user_profile, k=15)
    
    if not results:
        return "I'm sorry, No similar job description found in the database."
    
    # Format results
    jobs_data = "=== JOB POSITIONS FROM SIMILAR JOB DESCRIPTION ===\n\n"
    jobs_data += f"Found {len(results)} job vacancies with similar description in our database.\n"
    jobs_data += "Analyzing their career paths and current positions...\n\n"
    
    for i, result in enumerate(results, 1):
        jobs_data += f"--- Similar Description In The Database {i} ---\n"
        jobs_data += f"{result.page_content[:800]}\n\n"
    
    jobs_data += "Based on these profiles, consider the following for job recommendations:\n"
    jobs_data += "1. Most common job titles\n2. Industries\n3. Career progression patterns\n4. Required skills\n"
    
    return jobs_data

# ===== JOB RECOMMENDATION Agent =====
JOB_RECOMMENDATION_PROMPT = """
# ROLE
Career Advisor AI that recommends jobs by analyzing real indonesia job vacancy data from indonesian_job_v2 database inside Qdrant vectorstore.

**STRICT RULES**
1. NEVER invent job titles or companies that do not exist in the retrieval output.
2. ALWAYS call tools if additional job-matching is required.
3. DO NOT guess missing information.
4. Follow the formatting rules exactly.
5. Be concise, deterministic, and factual.
6. TO PERFORM THIS TASK, YOU MUST GET DATA FROM RAG AGENT AND DON'T HALLUCINATE.
7. ONLY use the 'job_recommendation' tool to get job recommendations.
8. ONLY answer question using previous history chat if the user asks about it with the exact words, for instances : "previous context", "history", "based on previous profile", "berdasarkan profil tersebut", etc. If NOT, DO NOT use previous history chat as reference!!!

# OUTPUT FORMAT
**💼 Career Recommendations**
"""


def recommendation_agent(question: str, history: str = None):
    """
    All queries routed to job recommendation agent.
    """
    agent = create_agent(
        model=llm, 
        tools=[job_recommendation])
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"{JOB_RECOMMENDATION_PROMPT}\n\n"
                       f"**IMPORTANT: Answer ONLY the current question below.**\n\n"
                       f"**Current Question**: {question}\n"
                       f"**Previous Context**: {history}"
        }]
    })
    answer = result["messages"][-1].content
    return {
        "answer": answer,
        "selected_agent": "job_recommendation_agent",
        "tool_messages": "Job recommendation tool used."
    }
