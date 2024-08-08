# app/main.py
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests
from typing import Literal
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from app.templates import prompts
from langchain_core.tools import tool
import time
import json

load_dotenv()

SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_ENDPOINT = os.getenv('SLACK_ENDPOINT')
OPENAI_KEY = os.getenv("OPENAI_KEY")
TYPING_RACE_CHANNEL = os.getenv("TYPING_RACE_CHANNEL", "C07GS4G75R6")

app = FastAPI()

class SlackEvent(BaseModel):
    type: str
    event: dict

def get_bot_user_id():
    url = f"{SLACK_ENDPOINT}/auth.test"
    headers = {"Authorization": "Bearer " + SLACK_BOT_TOKEN}
    response = requests.post(url, headers=headers).json()
    return response.get('user_id')

BOT_USER_ID = get_bot_user_id()
typing_race_text = ""
race_start_time = 0
# In-memory store for processed event IDs
processed_event_ids = []
winners = []

@app.post("/newrace")
async def new_race(command: str = Form(...), text: str = Form(...), user_id: str = Form(...), channel_id: str = Form(...)):
    global typing_race_text
    global race_start_time
    global winners

    if channel_id != TYPING_RACE_CHANNEL:
        return {"status": "ignored"}

    typing_race_text = text
    race_start_time = time.time()
    winners = []
    post_message_to_slack(channel_id, "The race has started! 🟢")
    return "You got it! Starting a new race for the text: " + text

@app.post("/endrace")
async def end_race(command: str = Form(...), user_id: str = Form(...), channel_id: str = Form(...)):
    global typing_race_text
    global race_start_time
    global winners

    if channel_id != TYPING_RACE_CHANNEL:
        return {"status": "ignored"}

    typing_race_text = ""
    race_start_time = 0
    winners = []
    post_message_to_slack(channel_id, "The race has ended! 🔴")

    return 'Stopping the race!'

@app.post("/")
async def handle_slack_event(request: Request):
    body = await request.json()

    # Handle Slack URL verification challenge
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

    event = body.get("event", {})
    channel = event.get("channel")
    text = event.get("text")
    user = event.get("user")
    thread_ts = event.get("ts")
    event_id = body.get("event_id")

    print(f"Received event: {channel} - {text} - {user} - {thread_ts} - {event_id}")

    # Check if the event has already been processed
    if event_id in processed_event_ids:
        return {"status": "Event already processed"}

    processed_event_ids.append(event_id)

    if channel != TYPING_RACE_CHANNEL or user == BOT_USER_ID or text is None or typing_race_text == "":
        return {"status": "ignored"}
    
    success, response = process_message(text, user)
    if success:
        thread_ts = None
    if response:
        post_message_to_slack(channel, response, thread_ts)
    
    return {"status": "ok"}

def process_message(text: str, user: str):
    global typing_race_text
    global race_start_time
    global winners

    if not typing_race_text:
        return False, ""

    success, response = get_response(text)

    if success is None:
        return False, "Something went wrong. Please try again."

    if not success:
        return False, response
    else:
        end_time = time.time()
        elapsed_time = end_time - race_start_time
        medals = [":first_place_medal:", ":second_place_medal:", ":third_place_medal:"]
        winners.append((user, elapsed_time, medals[len(winners)]))

        if len(winners) < 3:
            return True, f"✅ <@{user}> completed the race in {elapsed_time:.2f} seconds! {medals[len(winners)-1]}\n{response}"
        else:
            typing_race_text = ""
            final_message = f"✅ <@{user}> completed the race in {elapsed_time:.2f} seconds! {medals[len(winners)-1]}\n{response}\n\n"
            final_message += "🏆 The winners are:\n"
            for winner in winners:
                final_message += f"{winner[2]} <@{winner[0]}> with a time of {winner[1]:.2f} seconds\n"
            return True, final_message

def post_message_to_slack(channel: str, text: str, thread_ts: str = None):
    url = f"{SLACK_ENDPOINT}/chat.postMessage"
    headers = {"Authorization": "Bearer " + SLACK_BOT_TOKEN}
    data = {
        "channel": channel,
        "text": text,
        "thread_ts": thread_ts
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

@tool
def highlight_error(error: str = None) -> str:
    """Brief description of the error in the provided text"""
    return False, error

@tool
def celebrate(celebration_message: str = None) -> str:
    """Used only if the input text matches the given text exactly"""
    return True, celebration_message

tools = [highlight_error, celebrate]
tool_node = ToolNode(tools)

# Define conditional edge
def should_continue(state: MessagesState) -> Literal["tools", 'end']:
    messages = state['messages']
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools"
    return "end"

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    model = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name="gpt-4o-mini")
    
    tool_llm = model.bind_tools(tools)
    response = tool_llm.invoke(messages)

    return {"messages": [response]}

def end(state: MessagesState):
    messages = state['messages']
    last_message = messages[-1]

    return last_message

# Define a new graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("end", end)
workflow.add_conditional_edges("agent", should_continue)
workflow.set_entry_point("agent")

def get_response(text: str):
    model = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name="gpt-4o-mini")
    prompt = prompts["system"].format(text=typing_race_text, user_input=text)
    tool_llm = model.bind_tools(tools)
    response = tool_llm.invoke(prompt)

    # Parse the response
    tool_calls = response.additional_kwargs.get('tool_calls', [])
    if not tool_calls:
        return None, None

    errors = []
    success = False
    message = ""

    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        arguments = json.loads(tool_call['function']['arguments'])  # Parse the arguments string as JSON
        if function_name == 'highlight_error':
            error_message = arguments.get('error', '')
            if error_message:
                errors.append(f"❌ {error_message}")
        elif function_name == 'celebrate':
            # Validate the strings manually
            if text.replace(" ", "") == typing_race_text.replace(" ", ""):
                success = True
                message = arguments.get('celebration_message', 'Congratulations! You typed the text correctly.')
            else:
                errors.append("❌ Close, but the strings don't match exactly.")
    
    if not success:
        message = "\n".join(errors)
    
    return (success, message)
