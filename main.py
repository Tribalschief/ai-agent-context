import os
from dotenv import load_dotenv
from typing import cast , List
import chainlit as cl
from agents import Agent , Runner , AsyncOpenAI , OpenAIChatCompletionsModel
from agents.run import RunConfig
from agents.tool import function_tool
from agents.run_context import RunContextWrapper

load_dotenv()

gemini_api_key = os.getenv("Google_Api")

if not gemini_api_key:
    raise ValueError("Google_Api environment variable is not set.")

@cl.set_starters
async def start() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Greeting",
            message="Hello! What can you help me with today?",
        ),
        cl.Starter(
            label="Weather",
            message="What is the weather like today?",
        )
    ]

class MyContext:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.seen_messages = []
        
@function_tool
@cl.step(type="weather tool")
def get_weather(location: str, unit: str = "C") -> str:
    """
    Fetch the weather for a given location, returning a short description.
    """
    # Example logic
    return f"The weather in {location} is 22 degrees {unit}."


@function_tool
@cl.step(type="greeting tool")
def greet_user(context: RunContextWrapper[MyContext], greeting: str) -> str:
    user_id = context.context.user_id
    return f"Hello {user_id}, you said: {greeting}"


@cl.on_chat_start
async def start():
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    )
    
    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=client,
    )
    
    config = RunConfig(
        model=model,
        model_provider=client,
        tracing_disabled=True,
    )
    
    """ Set up the chat session with the agent and context. """
    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)
    
    agent: Agent = Agent(
        name="Assistant",
        tools=[get_weather, greet_user],
        instructions="""You are a helpful assistant. You can answer questions about the weather and greet users. Use the tools provided to fetch weather information or greet the user based on their input.""",
        model=model,
    )
    
    cl.user_session.set("agent", agent)
    await cl.Message(
        content="Hello! What can I help you with today?",
    ).send()
    
@cl.on_message
async def main(message: cl.Message):
    """ Process incoming messages and generate responses"""
    
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    
    history = cl.user_session.get("chat_history") or []
    
    history.append({"role": "user", "content": message.content})
    
    my_ctx = MyContext(user_id="Ahmed")
    
    try:
        print("\n [CALLING AGENT WITH CONTEXT]\n", history, "\n")
        result = Runner.run_sync(agent, history, run_config=config, context=my_ctx)
        
        response_content = result.final_output
        
        msg.content = response_content
        await msg.update()
        
        history.append({"role": "assistant", "content": response_content})
        cl.user_session.set("chat_history", history)
        
        print("\n [FINAL RESPONSE]\n", response_content, "\n")
        print(f"User input: {message.content}")
        print(f"Response: {response_content}")
        
    except Exception as e:
        print(f"Error: {e}")
        await cl.Message(
            content=f"Error: {e}",
        ).send()
    