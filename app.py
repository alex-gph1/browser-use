import asyncio
import os
from fastapi import FastAPI, Body, HTTPException
import gradio as gr
from browser_use import Agent, BrowserProfile
from browser_use.llm import ChatOpenAI, ChatGoogle, ChatOpenRouter

app = FastAPI(title="MedsGo Browser Agent Hub")

# --- Model Factory ---
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
            return ChatOpenRouter(model=model, api_key=key, http_referer="https://medsgo.ph")
        elif provider == "z.ai":
            key = os.getenv("ZAI_API_KEY")
            if not key: raise ValueError("ZAI_API_KEY missing")
            return ChatOpenAI(model=model, api_key=key, base_url="https://api.z.ai/v1")
        
        # Default fallback
        return ChatGoogle(model="gemini-flash-lite-latest")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM Config Error: {str(e)}")

# --- API Endpoint ---
@app.post("/run")
async def run_task(
    task: str = Body(...), 
    provider: str = Body("google"), 
    model: str = Body("gemini-flash-lite-latest")
):
    llm = get_llm_model(provider, model)
    
    # Define the persistent profile for CS-Cart logins
    # This stores cookies in the CapRover /data volume
    profile = BrowserProfile(
        user_data_dir="/data/profiles/default"
    )

    agent = Agent(
        task=task,
        llm=llm,
        use_vision=True,
        browser_profile=profile
    )
    
    history = await agent.run()
    
    return {
        "status": "success",
        "final_result": history.final_result(),
        "steps_taken": len(history.history),
        "urls_visited": list(set(history.urls())) # Scannability: see where it went
    }

# --- Gradio UI ---
def create_ui():
    with gr.Blocks(title="MedsGo Agent HQ", theme=gr.themes.Soft()) as ui:
        gr.Markdown("# 🤖 MedsGo Browser Agent Hub")
        with gr.Row():
            provider_drop = gr.Dropdown(choices=["openai", "google", "openrouter", "z.ai"], label="Provider", value="google")
            model_input = gr.Textbox(label="Model Name", value="gemini-flash-lite-latest")
        
        task_input = gr.Textbox(label="Task Description", lines=4, placeholder="e.g. Go to medsgo.ph/admin and check pending orders...")
        
        with gr.Row():
            run_btn = gr.Button("🚀 Execute Task", variant="primary")
            
        output = gr.JSON(label="Execution Summary")

        run_btn.click(
            fn=lambda t, p, m: asyncio.run(run_task(t, p, m)),
            inputs=[task_input, provider_drop, model_input],
            outputs=output
        )
    return ui

app = gr.mount_gradio_app(app, create_ui(), path="/ui")
