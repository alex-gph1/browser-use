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

# --- 1. ARTIFACT & ENVIRONMENT SETUP ---
ARTIFACT_DIR = "/data/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Mount the folder so we can access files via URL
app.mount("/view", StaticFiles(directory=ARTIFACT_DIR), name="view")

# Fetch the public URL from the environment, defaulting to localhost for local dev
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:80").rstrip("/")


# --- 2. MODEL FACTORY ---
def get_llm_model(provider: str, model: str):
    provider = provider.lower()
    try:
        if provider == "google": 
            return ChatGoogle(model=model)
        elif provider == "openai": 
            return ChatOpenAI(model=model)
        elif provider == "openrouter":
            key = os.getenv("OPENROUTER_API_KEY")
            if not key: raise ValueError("OPENROUTER_API_KEY missing")
            return ChatOpenRouter(model=model, api_key=key)
        elif provider == "z.ai":
            key = os.getenv("ZAI_API_KEY")
            if not key: raise ValueError("ZAI_API_KEY missing")
            return ChatOpenAI(model=model, api_key=key, base_url="https://api.z.ai/v1")
        
        # Default fallback to save costs on simple tasks
        return ChatGoogle(model="gemini-flash-lite-latest")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM Config Error: {str(e)}")


# --- 3. CORE AGENT LOGIC ---
async def execute_agent(task: str, provider: str, model: str):
    """Core function used by both API and UI"""
    llm = get_llm_model(provider, model)
    
    # Persistent profile for CS-Cart logins
    profile = BrowserProfile(user_data_dir="/data/profiles/default")

    agent = Agent(
        task=task,
        llm=llm,
        use_vision=True,
        browser_profile=profile
    )
    
    history = await agent.run()
    
    # --- AUTOMATIC SCREENSHOT EXTRACTION ---
    final_screenshot_url = None
    
    # browser-use saves the state at each step. We grab the absolute last one.
    if history.history and history.history[-1].state.screenshot:
        filename = f"screenshot_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(ARTIFACT_DIR, filename)
        
        # Decode base64 and save to the persistent volume
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(history.history[-1].state.screenshot))
        
        # Construct the dynamic URL using your CapRover domain
        final_screenshot_url = f"{PUBLIC_URL}/view/{filename}"

    return {
        "status": "success",
        "final_result": history.final_result(),
        "final_screenshot": final_screenshot_url,
        "steps_taken": len(history.history)
    }


# --- 4. API ENDPOINT (For n8n / Jules) ---
@app.post("/run")
async def run_task_api(
    task: str = Body(...), 
    provider: str = Body("google"), 
    model: str = Body("gemini-flash-lite-latest")
):
    return await execute_agent(task, provider, model)


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
