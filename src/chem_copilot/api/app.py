"""FastAPI application for the Computational Chemistry Copilot.

Provides a REST API for:
  - Chatting with the agent (with tool calling)
  - Streaming responses via Server-Sent Events
  - Listing and querying Slurm jobs

Usage:
    uvicorn chem_copilot.api.app:app --host 0.0.0.0 --port 8000

Environment variables:
    OLLAMA_BASE_URL     — Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL        — Model name (default: qwen2.5-coder:7b)
    ANTHROPIC_API_KEY   — For Claude backend (optional)
    LLM_BACKEND         — "ollama" or "claude" (default: ollama)
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chem_copilot.agent import build_agent
from chem_copilot.llm import get_llm
from chem_copilot.slurm import SlurmManager

# ---------------------------------------------------------------------------
# App state (initialized on startup)
# ---------------------------------------------------------------------------

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize LLM and agent on startup."""
    backend = os.environ.get("LLM_BACKEND", "ollama")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
    llm = get_llm(backend=backend, model=model)
    _state["agent"] = build_agent(llm)
    _state["slurm"] = SlurmManager()
    yield
    _state.clear()


app = FastAPI(
    title="Chem Copilot",
    description="AI copilot for computational chemistry workflows",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """A user message to the agent."""
    message: str = Field(description="User's natural language query")
    thread_id: str | None = Field(
        default=None,
        description="Conversation thread ID for memory. Auto-generated if omitted.",
    )


class ChatResponse(BaseModel):
    """Agent response."""
    response: str = Field(description="Agent's text response")
    thread_id: str = Field(description="Thread ID for follow-up messages")
    tool_calls: list[dict] = Field(
        default_factory=list,
        description="Tools that were called during this turn",
    )


class JobStatus(BaseModel):
    """Slurm job status."""
    job_id: str
    state: str
    name: str = ""
    node: str = ""
    elapsed: str = ""
    job_dir: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the Chem Copilot agent.

    The agent will reason about the request, optionally call tools
    (input generators, job submission, output parsers), and return
    a response.
    """
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(503, "Agent not initialized")

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [("user", req.message)]},
        config=config,
    )

    # Extract response and tool calls
    messages = result["messages"]
    tool_calls_made = []
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls_made.append({
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                })

    final_msg = messages[-1]
    response_text = getattr(final_msg, "content", "") or ""

    return ChatResponse(
        response=response_text,
        thread_id=thread_id,
        tool_calls=tool_calls_made,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream agent response via Server-Sent Events.

    Each event contains a JSON object with 'type' (agent/tool) and 'content'.
    """
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(503, "Agent not initialized")

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator() -> AsyncGenerator[str, None]:
        import json

        for event in agent.stream(
            {"messages": [("user", req.message)]},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            data = json.dumps({
                                "type": "tool_call",
                                "name": tc["name"],
                            })
                            yield f"data: {data}\n\n"
                    elif getattr(msg, "type", None) == "tool":
                        content = msg.content if hasattr(msg, "content") else ""
                        data = json.dumps({
                            "type": "tool_result",
                            "content": content[:1000],
                        })
                        yield f"data: {data}\n\n"
                    elif hasattr(msg, "content") and msg.content:
                        data = json.dumps({
                            "type": "response",
                            "content": msg.content,
                            "thread_id": thread_id,
                        })
                        yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )


@app.get("/jobs", response_model=list[JobStatus])
async def list_jobs(n: int = 10) -> list[JobStatus]:
    """List recent Slurm jobs."""
    mgr: SlurmManager = _state.get("slurm")
    if mgr is None:
        raise HTTPException(503, "Slurm manager not initialized")

    jobs = mgr.list_recent_jobs(n=n)
    return [
        JobStatus(
            job_id=j.job_id, state=j.state, name=j.name,
            node=j.node, elapsed=j.elapsed, job_dir=j.job_dir,
        )
        for j in jobs
    ]


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Get status of a specific Slurm job."""
    mgr: SlurmManager = _state.get("slurm")
    if mgr is None:
        raise HTTPException(503, "Slurm manager not initialized")

    j = mgr.get_job_status(job_id)
    return JobStatus(
        job_id=j.job_id, state=j.state, name=j.name,
        node=j.node, elapsed=j.elapsed, job_dir=j.job_dir,
    )


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {
        "status": "ok",
        "agent_ready": "agent" in _state,
        "tools": 9,
    }
