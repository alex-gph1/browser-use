import asyncio
import base64
import os
import uuid
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

def get_llm_model(provider: str, model: str):
    provider = provider.lower()
    try:
        if provider == "google": return ChatGoogle(model=model)
        elif provider == "openai": return ChatOpenAI(model=model)
        elif provider == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key: raise ValueError("OPENROUTER_API_KEY missing")
            return ChatOpenRouter(model=model, api_key=key)
        elif provider == "z.ai":
            key = os.getenv("ZAI_API_KEY")
            if not key: raise ValueError("ZAI_API_KEY missing")
            return ChatOpenAI(model=model, api_key=key, base_url="https://api.z.ai/v1")
        return ChatGoogle(model="gemini-flash-lite-latest")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM Config Error: {str(e)}")

# --- BACKGROUND TASK ---
async def background_runner(job_id: str, task: str, provider: str, model: str):
    try:
        active_jobs[job_id]["status"] = "running"
        llm = get_llm_model(provider, model)
        profile = BrowserProfile(user_data_dir="/data/profiles/default")

        agent = Agent(
            task=task,
            llm=llm,
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
    provider: str = Body("google"), 
    model: str = Body("gemini-flash-lite-latest")
):
    job_id = uuid.uuid4().hex
    active_jobs[job_id] = {"status": "pending"}
    
    # Create an asyncio task and store its reference to allow cancellation
    job_task = asyncio.create_task(background_runner(job_id, task, provider, model))
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
                value="google"
            )
            model_input = gr.Textbox(
                label="Model Name", 
                value="gemini-flash-lite-latest"
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
            fn=lambda t, p, m: asyncio.run(execute_agent(t, p, m)),
            inputs=[task_input, provider_drop, model_input],
            outputs=output
        )
    return ui

# Mount the UI to /ui
app = gr.mount_gradio_app(app, create_ui(), path="/ui")
