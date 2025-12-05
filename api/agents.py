import re
from typing import Optional, List, Dict

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

# Chat function that acts as the MAIN AGENT
def chat(req: schema.ChatRequest, cv_file_contents: Optional[str]) -> str:
    """
    This is the main agent that routes questions to the appropriate specialist agent.
    """
    user_question = req.message.content
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

    # --- Agent States & Multi-turn Conversation ---
    # Check if the last message was the AI asking for a job description
    is_waiting_for_jd = False
    if req.history:
        last_message = req.history[-1]
        if last_message.role == 'ai' and "please paste the job description" in last_message.content.lower():
            is_waiting_for_jd = True

    # If waiting for a job description, the user's message is the job description.
    # Bypass the router and go straight to the ResumeAgent.
    if is_waiting_for_jd:
        if not cv_file_contents:
            return "Error: CV content is missing. Please upload your CV again before pasting the job description."
        
        print(f"[{req.session_id}] --- Executing Resume Analysis ---")
        resume_agent = ResumeAgent(llm)
        job_description = user_question
        analysis_result = resume_agent.run(cv_text=cv_file_contents, job_description=job_description)
        
        if "error" in analysis_result:
            return analysis_result["error"]
            
        # Format the response
        response = f"""
### Resume Analysis Complete

Here is your resume optimization report:

**ATS Score Analysis:**
- **Before:** {analysis_result['before_score']}/100
- **After:** {analysis_result['after_score']}/100

---

### Rewritten Resume (Optimized for ATS & Recruiters)

{analysis_result['rewritten_cv']}
"""
        return response.strip()

    # 1. Router: Decide which agent to use
    router_prompt_template = """
    You are an expert query router. Your job is to determine whether a user's question should be answered by a RAG agent, a SQL agent, or a Resume agent.
    - The RAG agent handles semantic questions about job descriptions, responsibilities, skills, and qualifications.
    - The SQL agent handles factual questions that can be answered from a database table with columns like 'work_type', 'salary', 'location', 'company_name', and 'job_title'.
    - The Resume agent handles requests to analyze, score, or rewrite a user's resume based on a job description. Look for keywords like 'resume', 'CV', 'optimize', 'rewrite', 'ATS'.

    Based on the user's question below, respond with "RAG", "SQL", or "RESUME".

    User Question: "{question}"
    """
    router_prompt = PromptTemplate.from_template(router_prompt_template)
    router_chain = LLMChain(llm=llm, prompt=router_prompt)

    # Placeholder for the RAG agent's response
    rag_response = "I am the RAG agent. This feature is being integrated."

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
        elif "resume" in route.lower():
            print(f"[{req.session_id}] --- Activating Resume Agent (Initial Request) ---")
            if not cv_file_contents:
                return "It looks like you want to work on your resume. Please upload your CV first using the button on the sidebar."
            # This is the prompt that sets up the next turn
            return "Resume agent is active. Please paste the job description you are targeting, and I will begin the analysis."
        else:
            print(f"[{req.session_id}] --- Activating RAG Agent (Placeholder) ---")
            return rag_response

    except Exception as e:
        print(f"[{req.session_id}] An error occurred in the main agent: {e}")
        return "Sorry, I encountered an error. Please try rephrasing your question."

class ResumeAgent:
    """
    An agent dedicated to optimizing resumes for ATS and human recruiters.
    """
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.action_verbs = [
            'achieved', 'accelerated', 'administered', 'advised', 'advocated', 'analyzed', 'authored', 'automated',
            'balanced', 'boosted', 'budgeted', 'built', 'calculated', 'centralized', 'chaired', 'clarified',
            'collaborated', 'conceived', 'conceptualized', 'coordinated', 'created', 'customized', 'decreased',
            'defined', 'designed', 'developed', 'directed', 'doubled', 'drove', 'eliminated', 'enabled', 'enforced',
            'engineered', 'enhanced', 'established', 'evaluated', 'executed', 'expanded', 'facilitated', 'formalized',
            'formed', 'formulated', 'founded', 'generated', 'grew', 'halved', 'headed', 'identified', 'implemented',
            'improved', 'increased', 'initiated', 'inspired', 'instituted', 'integrated', 'introduced', 'invented',
            'launched', 'led', 'lowered', 'managed', 'mastered', 'maximized', 'measured', 'mentored', 'minimized',
            'modernized', 'motivated', 'negotiated', 'optimized', 'orchestrated', 'overhauled', 'oversaw', 'pioneered',
            'planned', 'predicted', 'prepared', 'presented', 'prioritized', 'produced', 'proposed', 'proved',
            'provided', 'raised', 'rebuilt', 'recommended', 'redesigned', 'reduced', 're-engineered', 'refined',
            'reformed', 'reorganized', 'replaced', 'restructured', 'revamped', 'saved', 'scaled', 'scheduled',
            'secured', 'selected', 'simplified', 'slashed', 'solved', 'spearheaded', 'specified', 'standardized',
            'streamlined', 'strengthened', 'structured', 'succeeded', 'supervised', 'surpassed', 'systematized',

            'targeted', 'taught', 'tested', 'trained', 'transformed', 'tripled', 'unified', 'updated', 'upgraded',
            'validated', 'verbalized', 'verified', 'visualized', 'won'
        ]

    def _extract_keywords_from_jd(self, job_description: str) -> List[str]:
        """Extracts keywords from a job description using an LLM."""
        print("ResumeAgent: Extracting keywords from JD...")
        prompt_template = """
        You are an expert HR analyst. Your task is to extract the most important keywords from a job description.
        Focus on specific, marketable skills, technologies, and essential qualifications. Ignore generic phrases.
        Please return a simple comma-separated list of the top 10-15 keywords.

        Example:
        Job Description: "We are looking for a Senior Software Engineer with experience in Python, Django, and React. The ideal candidate will have a strong understanding of cloud services like AWS and a background in CI/CD pipelines."
        Output: "Python, Django, React, AWS, CI/CD, Senior Software Engineer"

        Job Description:
        "{job_description}"

        Output:
        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            result = chain.run(job_description=job_description)
            keywords = [keyword.strip() for keyword in result.split(',') if keyword.strip()]
            print(f"ResumeAgent: Extracted keywords: {keywords}")
            return keywords
        except Exception as e:
            print(f"ResumeAgent: Error extracting keywords: {e}")
            return []

    def _calculate_ats_score(self, cv_text: str, jd_keywords: List[str]) -> int:
        """Calculates a heuristic-based ATS score out of 100."""
        print("ResumeAgent: Calculating ATS score...")
        score = 0
        
        # 1. Keyword Matching (70 points)
        if jd_keywords:
            matches = 0
            cv_lower = cv_text.lower()
            for keyword in jd_keywords:
                if keyword.lower() in cv_lower:
                    matches += 1
            keyword_score = (matches / len(jd_keywords)) * 70
            score += keyword_score
            print(f"ResumeAgent: Keyword score: {keyword_score:.2f}/70")

        # 2. Quantitative Results (20 points)
        # Finds numbers, percentages, dollar amounts
        quant_results = re.findall(r'(\d+%|\$\d+|\d{2,})', cv_text)
        quant_score = min(len(quant_results) * 2, 20)
        score += quant_score
        print(f"ResumeAgent: Quantitative score: {quant_score}/20")

        # 3. Action Verbs (10 points)
        action_verb_matches = 0
        cv_words = set(re.findall(r'\b\w+\b', cv_text.lower()))
        found_verbs = {verb for verb in self.action_verbs if verb in cv_words}
        action_verb_score = min(len(found_verbs), 10)
        score += action_verb_score
        print(f"ResumeAgent: Action verb score: {action_verb_score}/10")
        
        final_score = min(int(score), 100)
        print(f"ResumeAgent: Final Score: {final_score}/100")
        return final_score

    def _rewrite_resume_with_llm(self, cv_text: str, jd_keywords: List[str]) -> str:
        """Rewrites the resume to be more impactful and keyword-rich."""
        print("ResumeAgent: Rewriting resume with LLM...")
        
        prompt_template = """
        You are a world-class career coach and professional resume writer, an expert at optimizing resumes for both Applicant Tracking Systems (ATS) and human recruiters.

        Your task is to rewrite the provided resume to be significantly more impactful and tailored for a specific job description.

        Here is the original resume:
        ---
        {cv_text}
        ---

        Here is a list of the most important keywords extracted from the target job description:
        ---
        {keywords}
        ---

        Follow these instructions precisely:
        1.  **Integrate Keywords:** Naturally and seamlessly weave the provided keywords into the resume. Do not just list them. They should fit the context of the experience and skills sections.
        2.  **Quantify Impact:** This is your most important task. Transform experience bullet points from simple duties into powerful, measurable achievements. Use metrics, numbers, percentages, and dollar amounts to show impact.
        3.  **Use Placeholders for Missing Numbers:** If the original resume lacks numbers, invent realistic and specific placeholders and enclose them in square brackets. For example, rewrite "Managed a team" to "Managed a team of [5] engineers" or "Improved performance" to "Improved system performance by [~20%]". This is crucial for creating a high-impact resume.
        4.  **Action Verbs:** Start every bullet point with a strong action verb.
        5.  **Preserve Structure:** Maintain the original section layout of the resume (e.g., Summary, Experience, Education, Skills).
        6.  **Return Only the Resume:** Your final output should ONLY be the full text of the rewritten, optimized resume. Do not include any commentary, greetings, or explanations before or after the resume text.

        Now, begin the rewrite.
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        try:
            # Convert list of keywords to a comma-separated string for the prompt
            keywords_str = ", ".join(jd_keywords)
            rewritten_cv = chain.run(cv_text=cv_text, keywords=keywords_str)
            print("ResumeAgent: Successfully rewrote resume.")
            return rewritten_cv
        except Exception as e:
            print(f"ResumeAgent: Error rewriting resume: {e}")
            return "Error: Could not rewrite the resume at this time."

    def run(self, cv_text: str, job_description: str) -> Dict[str, any]:
        """
        Runs the full resume analysis and rewrite workflow.
        """
        print("ResumeAgent: Starting analysis...")
        
        # 1. Extract keywords from the job description
        jd_keywords = self._extract_keywords_from_jd(job_description)
        
        if not jd_keywords:
            return {
                "error": "Could not extract keywords from the job description. Please try again with a different one."
            }
        
        # 2. Calculate the "before" score
        before_score = self._calculate_ats_score(cv_text, jd_keywords)
        
        # 3. Rewrite the resume
        rewritten_cv = self._rewrite_resume_with_llm(cv_text, jd_keywords)
        
        # 4. Calculate the "after" score
        after_score = self._calculate_ats_score(rewritten_cv, jd_keywords) # No longer a placeholder increase
        
        print("ResumeAgent: Analysis complete.")
        
        return {
            "before_score": before_score,
            "after_score": after_score,
            "rewritten_cv": rewritten_cv,
        }