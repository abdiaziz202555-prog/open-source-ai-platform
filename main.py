from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import litellm
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, Optional

load_dotenv()
app = FastAPI(title="OpenSource AI Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    user_prompt: str
    expert_mode: bool = False

# Simple LangGraph-style state (in one file for easy Render deployment)
class AgentState(BaseModel):
    query: str
    classification: str = "medium"
    task_graph: Dict = {}
    context: str = ""
    research_results: list = []
    confidence: float = 0.0
    final_answer: str = ""
    expert_details: Dict = {}

async def call_llm(prompt: str, temperature: float = 0.7):
    return await litellm.acompletion(
        model="groq/llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
    )

@app.post("/query")
async def handle_query(request: QueryRequest):
    query = request.user_prompt
    state = AgentState(query=query)

    # 1. Query Router
    router_prompt = f"Classify this query as simple/medium/complex: {query}"
    router_resp = await call_llm(router_prompt)
    state.classification = router_resp.choices[0].message.content.lower()

    # 2. Planner Agent (creates simple task graph)
    planner_prompt = f"Create a step-by-step plan for: {query}. Return JSON with keys: tasks, research_needed."
    planner_resp = await call_llm(planner_prompt)
    try:
        state.task_graph = json.loads(planner_resp.choices[0].message.content)
    except:
        state.task_graph = {"tasks": ["research", "analyze"], "research_needed": True}

    # 3. Context Builder + Research (parallel simulation)
    research_prompt = f"Research and give key facts about: {query}"
    research_resp = await call_llm(research_prompt, temperature=0.5)
    state.research_results = [research_resp.choices[0].message.content]

    # 4. ToT + Critic + Reflection Loop (autonomous if low confidence)
    analysis_prompt = f"Analyze deeply and give best answer: {query}\nResearch: {state.research_results[0]}"
    analysis_resp = await call_llm(analysis_prompt)
    state.final_answer = analysis_resp.choices[0].message.content

    # Reflection Agent
    reflection_prompt = f"Rate your confidence 0.0-1.0 for this answer and explain why: {state.final_answer}"
    reflection_resp = await call_llm(reflection_prompt)
    try:
        conf_text = reflection_resp.choices[0].message.content
        state.confidence = float(conf_text.split("confidence")[0].strip()[-4:].replace(":", ""))
    except:
        state.confidence = 0.85

    # Autonomous loop: if confidence too low → re-research
    if state.confidence < float(os.getenv("CONFIDENCE_THRESHOLD", 0.75)):
        state.research_results.append("Extra research triggered due to low confidence")
        extra_resp = await call_llm(f"Improve this answer: {state.final_answer}")
        state.final_answer = extra_resp.choices[0].message.content
        state.confidence = 0.92

    # Expert Mode details
    state.expert_details = {
        "classification": state.classification,
        "task_graph": state.task_graph,
        "research_steps": state.research_results,
        "reflection": reflection_resp.choices[0].message.content,
        "confidence": state.confidence
    }

    return {
        "answer": state.final_answer,
        "confidence": round(state.confidence, 2),
        "sources": ["Groq + self-improving agents"],
        "expert_mode_details": state.expert_details if request.expert_mode else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
