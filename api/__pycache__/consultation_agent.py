import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.tools import tool
from langchain.agents import create_agent

# ===== SECRET KEY =====
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 


# ===== llm model setup =====
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

# Consultation tool 
@tool
def consultation(user_profile: str) -> str:
    """
    Advanced career consultation tool for providing full-spectrum career consultation including :
    - Job recommendation based on user profile
    - Skill-gap analysis and development plan
    - Career path visualization and progression insights
    - Resume & cover letter suggestions
    - Interview preparation tips
    - Industry trend insights and salary benchmarking
    - Personalized job search strategies and networking advice 
    - Use "indonesian_job_v2" Qdrant collection and LLM knowledge if needed to supplement courses and certifications recommendations
    """
    results = qdrant.similarity_search(user_profile, k=5)

    if not results :
        return "I'm sorry, No similar job description found in the database."
    
    #Report header
    report = "=== CAREER CONSULTATION REPORT ===\n\n"
    report += f"Analyzed {len(results)} similar job listings in the database.\n\n"

    #job insights
    report += "JOB RECOMMENDATIONS:\n"
    for i, result in enumerate(results, 1):
        report += f"{i}. {result.metadata.get('title', 'N/A')} at {result.metadata.get('company', 'N/A')} - {result.metadata.get('location', 'N/A')}\n"
    report += "\n"
    
    #skill-gap analysis
    report += "SKILL-GAP ANALYSIS:\n"
    report += "- Identify skills you you currently lack based on job requirements in similar roles.\n"
    report += "- Recommend online courses, certifications, or training programs to bridge these gaps.\n\n"

    #career path insights
    report += "CAREER PATH INSIGHTS:\n"
    report += "- Visualize potential career trajectories based on similar profiles.\n"
    report += "- Highlight common progression patterns and milestones in your desired field.\n\n"

    #resume suggestions
    report += "RESUME & COVER LETTER SUGGESTIONS:\n"
    report += "- Highlight the most relevant skills and experiences that align with the job descriptions found.\n\n"
    report += "- Tips for ATS optimization and tailored content for target roles.\n\n"

    #interview preparation
    report += "INTERVIEW PREPARATION TIPS:\n"
    report += "- Common interview questions for the roles identified.\n"
    report += "- Strategies for effective responses and showcasing your strengths.\n\n"
    
    #industry trends
    report += "INDUSTRY TRENDS & SALARY BENCHMARKING:\n"
    report += "- Current trends and in-demand skills in your target industry based on job listings.\n"
    report += "- Salary benchmarks based on role and experience.\n\n"

    #job search strategies
    report += "PERSONALIZED JOB SEARCH STRATEGIES & NETWORKING ADVICE:\n"
    report += "- Platforms to focus on, networking tips, and application prioritization strategies.\n\n"

    return report 

consultation_prompt = """
# ROLE
You are an Advanced Career Consultant AI designed to provide **full-spectrum, actionable, and in-depth career guidance**.
Your responses must be based primarily on real job vacancy data from the "indonesian_job_v2" Qdrant vectorstore.
You can supplement your advice with general LLM knowledge for recommendations on courses, certifications, or career strategies, 
but NEVER invent job titles, companies, or details not found in the retrieval output.

Your task is to dynamically generate a detailed Career Consultation Report tailored to the user's brief profile, CV/Resume, or questions.
The report should explain and justify each recommendation and insight, providing actionable advice for career advancement.

# EXPECTED INPUT
- User profile information including:
  - Name, education, and degree
  - Work experience, positions, and responsibilities
  - Skills, both technical and soft skills
  - Career goals and preferred industries or locations
- Specific questions or goals from the user (optional), e.g.:
  - "Which roles are best for my profile?"
  - "What skills should I acquire next?"
  - "How can I prepare for interviews in AI engineering?"

# REPORT SEGMENTS (must be included in every output)
1. **Job Recommendations**  
   - List relevant job titles, companies, and locations from the database.
   - Provide rationale based on skills, experience, and career goals.

2. **Skill-Gap Analysis**  
   - Identify missing skills relative to similar job postings.
   - Suggest concrete steps: courses, certifications, or hands-on practice.

3. **Career Path Insights** 
   - Show realistic potential career trajectories and progression milestones.
   - Include typical timelines and common patterns observed in the database.
   **REMEMBER** : YOU NEED TO ANALYZE THE USER'S PROFILE AND THE RETRIEVED SIMILAR JOB DESCRIPTIONS TO PROVIDE DETAILED AND DYNAMIC CAREER PATH INSIGHTS!

4. **Resume & Cover Letter Suggestions**  
   - Tips for highlighting relevant skills and achievements.
   - Guidance for tailoring content to ATS-friendly formats.

5. **Interview Preparation Tips**  
   - Common technical and behavioral questions for recommended roles.
   - Strategies for structuring answers and showcasing strengths.

6. **Industry Trends & Salary Benchmarking**  
   - Current in-demand skills and emerging roles in the target industry.
   - Salary range guidance based on experience and position.

7. **Personalized Job Search Strategies & Networking Advice**  
   - Recommended job platforms, networking approaches, and application prioritization.
   - Tips for leveraging connections, alumni networks, or online communities.

# STRICT RULES
1. ALWAYS retrieve job-specific data from the 'consultation' tool.
2. DO NOT hallucinate job titles, companies, or role details.
3. Only supplement skill, course, or certification recommendations with LLM knowledge.
4. Be **dynamic and detailed**: explain the reasoning behind every recommendation.
5. Be concise but informative: each segment should provide actionable insights.
6. Follow the **report segments** exactly; do not omit any.
7. ONLY reference previous context if the user explicitly asks for "previous context" or "history".

# OUTPUT FORMAT
**📝 Career Consultation Report**

- Each segment (Job Recommendations, Skill-Gap Analysis, Career Path Insights, etc.) must be included.
- Explain the rationale behind each recommendation.
- Explain detail in each section.
- Provide examples, reasoning, and actionable advice in each section.
- Output should read naturally as a professional career consultation report.
"""

def consultation_agent(question: str, history: str = None):
    """
    Routes all advanced career-related queries to consultation agent
    """
    consultation_agent = create_agent(
        model=llm,
        tools=[consultation])
    result = consultation_agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"{consultation_prompt}\n\n"
                       f"**IMPORTANT: Answer ONLY the current question below.**\n\n"
                       f"**Current Question**: {question}\n"
                       f"**Previous Context**: {history}"}]
    })

    answer = result["messages"][-1].content
    return {
        "answer": answer,
        "selected_agent": "consultation_agent",
        "tool_messages": "Consultation tool used."
    }



