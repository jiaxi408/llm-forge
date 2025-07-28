import asyncio

from langchain_sandbox import PyodideSandbox

from langgraph.checkpoint.memory import MemorySaver
from codeact import create_codeact
from codeact.eval_fn import create_pyodide_eval_fn

from langchain_openai import ChatOpenAI

# Define your tool function
def derivative(f, x, h=1e-5):
    """
    Compute the numerical derivative of function f at point x using central difference.

    Parameters:
    - f: callable, the function to differentiate
    - x: float, the point at which to evaluate the derivative
    - h: float, optional, the step size for the finite difference (default: 1e-5)

    Returns:
    - float, the approximate derivative f'(x)
    """
    return (f(x + h) - f(x - h)) / (2 * h)


tools = [derivative]

model = ChatOpenAI(
    model="Shanghai_AI_Laboratory/internlm3-8b-instruct-awq",
    openai_api_base="http://localhost:23333/v1",
    api_key="EMPTY",
    temperature=0.8
)

sandbox = PyodideSandbox(allow_net=True)
eval_fn = create_pyodide_eval_fn(sandbox)
code_act = create_codeact(model, tools, eval_fn)
agent = code_act.compile(checkpointer=MemorySaver())

async def run_agent(messages: list, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    # Stream agent outputs
    print("🤖: ", end="", flush=True)
    # Python version 3.11 or above is required for astream to support streaming output
    async for typ, chunk in agent.astream(
        {"messages": messages},
        stream_mode=["values", "messages"],
        config=config,
    ):
        if typ == "messages":
            print(chunk[0].content, end="")

if __name__ == "__main__":
    messages = []
    user_input = input("🤔: ")
    messages.append({
        "role": "user",
        "content": user_input
    })
    asyncio.run(run_agent(messages, "1"))