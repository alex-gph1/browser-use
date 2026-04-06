import asyncio
import base64
import os
import uuid
from typing import Any

from fastapi import FastAPI, Body, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
import gradio as gr
from browser_use import Agent, BrowserProfile
from browser_use.llm import ChatOpenAI, ChatGoogle, ChatOpenRouter
from browser_use.tools.service import Tools

app = FastAPI(title="Browser Agent Hub")

# --- ARTIFACT & ENVIRONMENT SETUP ---
ARTIFACT_DIR = "/data/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
app.mount("/view", StaticFiles(directory=ARTIFACT_DIR), name="view")
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:80").rstrip("/")

# --- IN-MEMORY JOB STORE ---
active_jobs = {}

# --- LLM DEFAULTS/FALLBACKS (ENV-DRIVEN) ---
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "google")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-flash-lite-latest")
FALLBACK_LLM_PROVIDER = os.getenv("FALLBACK_LLM_PROVIDER")
FALLBACK_LLM_MODEL = os.getenv("FALLBACK_LLM_MODEL")
AGENT_MAX_FAILURES = int(os.getenv("AGENT_MAX_FAILURES", "8"))
AGENT_STEP_TIMEOUT = int(os.getenv("AGENT_STEP_TIMEOUT", "240"))
AGENT_LLM_TIMEOUT = int(os.getenv("AGENT_LLM_TIMEOUT", "120"))


def _build_llm(provider: str, model: str):
    provider = provider.lower()
    if provider == "google":
        return ChatGoogle(model=model)
    if provider == "openai":
        return ChatOpenAI(model=model)
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY missing")
        return ChatOpenRouter(model=model, api_key=key)
    if provider == "z.ai":
        key = os.getenv("ZAI_API_KEY")
        if not key:
            raise ValueError("ZAI_API_KEY missing")
        return ChatOpenAI(model=model, api_key=key, base_url="https://api.z.ai/v1")
    raise ValueError(f"Unsupported provider '{provider}'. Use one of: google, openai, openrouter, z.ai")


def get_llm_models(
    provider: str | None,
    model: str | None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
) -> tuple[Any, Any | None]:
    """Build primary/fallback LLMs using request params with env-based defaults."""
    primary_provider = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
    primary_model = (model or DEFAULT_LLM_MODEL).strip()
    fallback_provider_resolved = (fallback_provider or FALLBACK_LLM_PROVIDER or "").strip().lower()
    fallback_model_resolved = (fallback_model or FALLBACK_LLM_MODEL or "").strip()

    try:
        primary_llm = _build_llm(primary_provider, primary_model)
        fallback_llm = None
        if fallback_provider_resolved and fallback_model_resolved:
            fallback_llm = _build_llm(fallback_provider_resolved, fallback_model_resolved)
        elif fallback_provider_resolved or fallback_model_resolved:
            raise ValueError("Both fallback_provider and fallback_model must be set together.")
        return primary_llm, fallback_llm
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM Config Error: {str(e)}")

# --- BACKGROUND TASK ---
async def background_runner(
    job_id: str,
    task: str,
    provider: str | None,
    model: str | None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    device_view: str | None = "mobile",
    device_size: str | None = "medium",
    include_tools: list[str] | None = None,
    generate_gif: bool | None = False,
    include_attributes: list[str] | None = None,
):
    try:
        active_jobs[job_id]["status"] = "running"
        llm, fallback_llm = get_llm_models(provider, model, fallback_provider, fallback_model)
        
        # Viewport configuration
        viewports = {
            "mobile": {
                "low": {"width": 320, "height": 640},
                "medium": {"width": 390, "height": 844},
                "high": {"width": 430, "height": 932},
                "ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
            },
            "tablet": {
                "low": {"width": 768, "height": 1024},
                "medium": {"width": 810, "height": 1080},
                "high": {"width": 1024, "height": 1366},
                "ua": "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
            },
            "desktop": {
                "low": {"width": 1280, "height": 720},
                "medium": {"width": 1920, "height": 1080},
                "high": {"width": 2560, "height": 1440},
                "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            }
        }
        
        view_type = (device_view or "mobile").lower()
        if view_type not in viewports:
            view_type = "mobile"
            
        size_type = (device_size or "medium").lower()
        if size_type not in ["low", "medium", "high"]:
            size_type = "medium"
            
        view_config = viewports[view_type]
        size_config = view_config[size_type]
        viewport_size = {"width": size_config["width"], "height": size_config["height"]}
        
        profile = BrowserProfile(
            user_data_dir=f"/data/profiles/{job_id}",
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"],
            window_size=viewport_size,
            viewport=viewport_size,
            user_agent=view_config["ua"]
        )

        tools = None
        if include_tools is not None:
            temp_tools = Tools()
            all_actions = list(temp_tools.registry.registry.actions.keys())
            exclude_actions = [a for a in all_actions if a not in include_tools]
            tools = Tools(exclude_actions=exclude_actions)
            
        session_fs_path = os.path.join(ARTIFACT_DIR, job_id)
        os.makedirs(session_fs_path, exist_ok=True)
        
        gif_path = None
        if generate_gif:
            gif_filename = f"history_gif_{job_id}.gif"
            gif_path = os.path.join(ARTIFACT_DIR, gif_filename)

        agent = Agent(
            task=task,
            llm=llm,
            fallback_llm=fallback_llm,
            use_vision=True,
            max_failures=AGENT_MAX_FAILURES,
            step_timeout=AGENT_STEP_TIMEOUT,
            llm_timeout=AGENT_LLM_TIMEOUT,
            browser_profile=profile,
            tools=tools,
            file_system_path=session_fs_path,
            generate_gif=gif_path if generate_gif else False,
            include_attributes=include_attributes if include_attributes is not None else []
        )
        
        # We store the agent reference in case we ever want to implement live logs/steps later
        active_jobs[job_id]["agent"] = agent
        
        history = await agent.run()
        
        final_screenshot_url = None
        if history.history:
            last_step = history.history[-1]
            screenshot_b64 = None
            if hasattr(last_step.state, 'get_screenshot'):
                screenshot_b64 = last_step.state.get_screenshot()
            elif hasattr(last_step, 'result') and hasattr(last_step.result, 'screenshot'):
                screenshot_b64 = last_step.result.screenshot
            elif hasattr(last_step.state, 'screenshot'):
                screenshot_b64 = last_step.state.screenshot
                 
            if screenshot_b64:
                filename = f"screenshot_{uuid.uuid4().hex[:8]}.png"
                filepath = os.path.join(ARTIFACT_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(base64.b64decode(screenshot_b64))
                final_screenshot_url = f"{PUBLIC_URL}/view/{filename}"

        final_gif_url = None
        if gif_path and os.path.exists(gif_path):
            final_gif_url = f"{PUBLIC_URL}/view/{os.path.basename(gif_path)}"

        active_jobs[job_id].update({
            "status": "completed",
            "data": {
                "final_result": history.final_result(),
                "final_screenshot": final_screenshot_url,
                "final_gif": final_gif_url,
                "steps_taken": len(history.history)
            }
        })
        
    except asyncio.CancelledError:
        # This catches the specific signal sent by our /stop endpoint
        active_jobs[job_id]["status"] = "cancelled"
        # We catch and swallow the CancelledError so the server logs stay clean
    except Exception as e:
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })

# --- API ENDPOINTS ---

@app.get("/help", response_class=PlainTextResponse)
async def get_help():
    help_text = f"""# Browser Agent Hub API

## Overview
This API allows agents to request automated browser interactions.

## Base URL
{PUBLIC_URL}

## Endpoints

### 1. `POST /run`
Starts a new browser agent job.
**Request Body (JSON):**
- `task` (string, required): The task description for the agent.
- `provider` (string, optional): LLM provider (e.g., 'google', 'openai', 'openrouter', 'z.ai').
- `model` (string, optional): Specific LLM model to use.
- `fallback_provider` (string, optional): Fallback LLM provider.
- `fallback_model` (string, optional): Fallback LLM model.
- `device_view` (string, optional): Viewport device. One of 'mobile' (default), 'tablet', 'desktop'.
- `device_size` (string, optional): Viewport dimension constraint. One of 'low', 'medium' (default), 'high'.
- `include_tools` (list of strings, optional): Explicit list of tool names to enable. If omitted, all tools are enabled. File operations are strictly sandboxed per job.
- `generate_gif` (boolean, optional): Whether to generate a GIF of the agent's actions (default: false).
- `include_attributes` (list of strings, optional): HTML attributes to include in the DOM context passed to the model (e.g. `["data-testid", "class"]`).

**Response:**
Returns a JSON object with `job_id`, `status_url` to poll the job state, and `stop_url` to cancel.

### 2. `GET /status/{{job_id}}`
Retrieves the current status and output of a job.
**Response Data:**
Contains status (`pending`, `running`, `completed`, `failed`, `cancelled`), and `data` objects with `final_result`, `final_screenshot` URL, and `final_gif` URL if generated.

### 3. `POST /stop/{{job_id}}`
Cancels an active job. Returns success message or error if already complete.
"""
    return help_text

@app.post("/run")
async def run_task_api(
    task: str = Body(...), 
    provider: str | None = Body(None),
    model: str | None = Body(None),
    fallback_provider: str | None = Body(None),
    fallback_model: str | None = Body(None),
    device_view: str | None = Body("mobile"),
    device_size: str | None = Body("medium"),
    include_tools: list[str] | None = Body(None),
    generate_gif: bool | None = Body(False),
    include_attributes: list[str] | None = Body(None),
):
    job_id = uuid.uuid4().hex
    active_jobs[job_id] = {"status": "pending"}
    
    # Create an asyncio task and store its reference to allow cancellation
    job_task = asyncio.create_task(
        background_runner(
            job_id=job_id,
            task=task,
            provider=provider,
            model=model,
            fallback_provider=fallback_provider,
            fallback_model=fallback_model,
            device_view=device_view,
            device_size=device_size,
            include_tools=include_tools,
            generate_gif=generate_gif,
            include_attributes=include_attributes,
        )
    )
    active_jobs[job_id]["task_ref"] = job_task
    
    return {
        "job_id": job_id,
        "status": "pending",
        "status_url": f"{PUBLIC_URL}/status/{job_id}",
        "stop_url": f"{PUBLIC_URL}/stop/{job_id}" # NEW
    }

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = active_jobs[job_id]
    return {
        "status": job_info["status"],
        "data": job_info.get("data"),
        "error": job_info.get("error")
    }

@app.post("/stop/{job_id}")
async def stop_job(job_id: str):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = active_jobs[job_id]
    if job_info["status"] in ["completed", "failed", "cancelled"]:
        return {"status": job_info["status"], "message": "Job is no longer running."}
    
    task_ref = job_info.get("task_ref")
    if task_ref and not task_ref.done():
        task_ref.cancel() # This interrupts the agent and triggers the CancelledError
        return {"status": "success", "message": f"Cancellation requested for job {job_id}."}
        
    return {"status": "error", "message": "Could not cancel task."}

# --- 5. GRADIO UI (For Manual Oversight) ---

async def execute_agent(task: str, provider: str, model: str, fallback_provider: str | None = None, fallback_model: str | None = None):
    try:
        llm, fallback_llm = get_llm_models(provider, model, fallback_provider, fallback_model)
        
        run_id = uuid.uuid4().hex[:8]
        profile = BrowserProfile(
            user_data_dir=f"/data/profiles/ui_manual_{run_id}",
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
        )

        agent = Agent(
            task=task,
            llm=llm,
            fallback_llm=fallback_llm,
            use_vision=True,
            max_failures=AGENT_MAX_FAILURES,
            step_timeout=AGENT_STEP_TIMEOUT,
            llm_timeout=AGENT_LLM_TIMEOUT,
            browser_profile=profile
        )
        
        history = await agent.run()
        
        final_screenshot = None
        if history.history:
            last_step = history.history[-1]
            if hasattr(last_step.state, 'get_screenshot'):
                final_screenshot = "Screenshot captured but not saved to disk in UI mode."
        
        return {
            "status": "completed",
            "final_result": history.final_result(),
            "steps_taken": len(history.history),
            "final_screenshot": final_screenshot
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

def create_ui():
    with gr.Blocks(title="Agent HQ") as ui:
        gr.Markdown("# 🤖 Browser Agent Hub")
        
        with gr.Row():
            provider_drop = gr.Dropdown(
                choices=["google", "openai", "openrouter", "z.ai"], 
                label="Provider", 
                value=DEFAULT_LLM_PROVIDER
            )
            model_input = gr.Textbox(
                label="Model Name", 
                value=DEFAULT_LLM_MODEL
            )
            fallback_provider_drop = gr.Dropdown(
                choices=["", "google", "openai", "openrouter", "z.ai"],
                label="Fallback Provider (optional)",
                value=FALLBACK_LLM_PROVIDER or ""
            )
            fallback_model_input = gr.Textbox(
                label="Fallback Model Name (optional)",
                value=FALLBACK_LLM_MODEL or ""
            )
        
        task_input = gr.Textbox(
            label="Task Description", 
            lines=4, 
            placeholder="e.g. Go to example.com/admin and check pending orders..."
        )
        
        with gr.Row():
            run_btn = gr.Button("🚀 Execute Task", variant="primary")
            
        output = gr.JSON(label="Execution Summary")

        # Wire up the button to the core async function
        run_btn.click(
            fn=lambda t, p, m, fp, fm: asyncio.run(execute_agent(t, p, m, fp or None, fm or None)),
            inputs=[task_input, provider_drop, model_input, fallback_provider_drop, fallback_model_input],
            outputs=output
        )
    return ui

# Mount the UI to /ui
app = gr.mount_gradio_app(app, create_ui(), path="/ui")
