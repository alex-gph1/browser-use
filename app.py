import asyncio
import os
from fastapi import FastAPI, Body
import gradio as gr
from browser_use import Agent
from browser_use.llm import ChatOpenAI, ChatGoogle, ChatOpenRouter

app = FastAPI(title="MedsGo Browser Agent Hub")

# --- Model Factory ---
def get_llm_model(provider: str, model: str):
    provider = provider.lower()
    
    if provider == "google":
        return ChatGoogle(model=model)
    elif provider == "openai":
        return ChatOpenAI(model=model)
    elif provider == "openrouter":
        return ChatOpenRouter(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            http_referer="https://medsgo.ph"
        )
    elif provider == "z.ai":
        return ChatOpenAI(
            model=model,
            api_key=os.getenv("ZAI_API_KEY"),
            base_url="https://api.z.ai/v1"
        )
    return ChatGoogle(model="gemini-flash-lite-latest") # Fallback

# --- API Endpoint (For n8n / Jules) ---
@app.post("/run")
async def run_task(
    task: str = Body(...), 
    provider: str = Body("google"), 
    model: str = Body("gemini-flash-lite-latest")
):
    llm = get_llm_model(provider, model)
    agent = Agent(
        task=task,
        llm=llm,
        use_vision=True
    )
    history = await agent.run()
    return {
        "status": "success",
        "final_result": history.final_result(),
        "steps_taken": len(history.history)
    }

# --- Gradio UI (For Alex) ---
def create_ui():
    with gr.Blocks(title="MedsGo Agent HQ") as ui:
        gr.Markdown("# 🤖 MedsGo Browser Agent")
        with gr.Row():
            provider_drop = gr.Dropdown(choices=["openai", "google", "openrouter", "z.ai"], label="Provider", value="google")
            model_input = gr.Textbox(label="Model Name", value="gemini-flash-lite-latest")
        task_input = gr.Textbox(label="Task Description", lines=3, placeholder="e.g. Login to CS-Cart and...")
        output = gr.JSON(label="Agent Result")
        run_btn = gr.Button("Execute", variant="primary")

        run_btn.click(
            fn=lambda t, p, m: asyncio.run(run_task(t, p, m)),
            inputs=[task_input, provider_drop, model_input],
            outputs=output
        )
    return ui

# Mount the UI to /ui
app = gr.mount_gradio_app(app, create_ui(), path="/ui")
