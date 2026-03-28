import asyncio
import base64
import os
import uuid
from typing import Any

from fastapi import FastAPI, Body, HTTPException
from fastapi.staticfiles import StaticFiles
import gradio as gr
from browser_use import Agent, BrowserProfile
from browser_use.llm import ChatOpenAI, ChatGoogle, ChatOpenRouter

app = FastAPI(title="MedsGo Browser Agent Hub")

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
):
    try:
        active_jobs[job_id]["status"] = "running"
        llm, fallback_llm = get_llm_models(provider, model, fallback_provider, fallback_model)
        
        # Add the Docker-stabilizing arguments here!
        profile = BrowserProfile(
            user_data_dir="/data/profiles/default",
            args=["--disable-dev-shm-usage", "--no-sandbox", "--disable-gpu"]
        )

        agent = Agent(
            task=task,
            llm=llm,
            fallback_llm=fallback_llm,
            use_vision=True,
            browser_profile=profile
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

        active_jobs[job_id].update({
            "status": "completed",
            "data": {
                "final_result": history.final_result(),
                "final_screenshot": final_screenshot_url,
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
@app.post("/run")
async def run_task_api(
    task: str = Body(...), 
    provider: str | None = Body(None),
    model: str | None = Body(None),
    fallback_provider: str | None = Body(None),
    fallback_model: str | None = Body(None),
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
def create_ui():
    with gr.Blocks(title="MedsGo Agent HQ", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🤖 MedsGo Browser Agent Hub")
        
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
            placeholder="e.g. Go to medsgo.ph/admin and check pending orders..."
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
