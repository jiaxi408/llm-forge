import inspect
from typing import Any, Callable, Awaitable, Optional, TypeVar, Type, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool as create_tool
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from codeact.utils import extract_and_combine_codeblocks

EvalFunction = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]
EvalCoroutine = Callable[[str, dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]

class CodeActState(MessagesState):
    script: Optional[str]
    context: dict[str, Any]

StateSchema = TypeVar("StateSchema", bound=CodeActState)
StateSchemaType = Type[StateSchema]

def create_default_prompt(tools: list[StructuredTool], base_prompt: Optional[str] = None):
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You will be given a task to execute. You should output one of the following two formats:
- A Python code snippet that provides a solution or a step toward solving the task. Any output you want to extract from the code should be printed to the console. The code must be enclosed in a fenced code block.
- Direct text to be shown to the user, if you need to ask for information or provide the final answer.

In addition to the Python standard library, you may use the following functions,
You can use the following functions directly; there is no need to define them again, and you must not redefine them!
"""

    for tool in tools:
        prompt += f'''
def {tool.name}{str(inspect.signature(tool.func))}:
    """{tool.description}"""
    ...
'''

    prompt += """

All top-level variables executed in previous interactions can be referenced directly.

Reminder: You must write Python code snippets when calling tools. Include print statements if needed. Any output from your code blocks will be returned to you, so you don't need to worry about capturing results — rely on what is printed.

Note: You may see 'Loading' or 'Loaded' in the output returned from the code execution. These are logs from package imports in your code. You do not need to be concerned with them — only the formal program output after these messages matters.

Important: Every code block you write will be executed. Do not output extra code blocks unless absolutely necessary to answer the user's question. For example, do NOT write installation instructions like 'pip install ...', as they will be executed and may cause errors. To avoid unnecessary issues, write only the core program logic. Ideally, your response should include only one code block unless additional text is required to answer the user.
"""
    return prompt

def create_codeact(
    model: BaseChatModel,
    tools: Sequence[Union[StructuredTool, Callable]],
    eval_fn: Union[EvalFunction, EvalCoroutine],
    *,
    prompt: Optional[str] = None,
    state_schema: StateSchemaType = CodeActState,
) -> StateGraph:
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]

    if prompt is None:
        prompt = create_default_prompt(tools)

    tools_context = {tool.name: tool.func for tool in tools}

    def call_model(state: StateSchema) -> Command:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = model.invoke(messages)

        code = extract_and_combine_codeblocks(response.content)

        if code:
            return Command(goto="sandbox", update={"messages": [response], "script": code})
        else:
            return Command(update={"messages": [response], "script": None})
        
    if inspect.iscoroutinefunction(eval_fn):
        async def sandbox(state: StateSchema):
            existing_context = state.get("context", {})
            context = {**existing_context, **tools_context}
            output, new_vars = await eval_fn(state["script"], context)
            new_context = {**existing_context, **new_vars}
            print("\n\n******************************** System Message ********************************")
            print(f"\n{output}")
            print("********************************************************************************\n")
            return {
                "messages": [{"role": "user", "content": output}],
                "context": new_context,
            }
    else:
        def sandbox(state: StateSchema):
            existing_context = state.get("context", {})
            context = {**existing_context, **tools_context}
            output, new_vars = eval_fn(state["script"], context)
            new_context = {**existing_context, **new_vars}
            print("\n\n******************************** System Message ********************************")
            print(f"\n{output}")
            print("********************************************************************************\n")
            return {
                "messages": [{"role": "user", "content": output}],
                "context": new_context,
            }
        
    agent = StateGraph(state_schema)
    agent.add_node(call_model, destinations=(END, "sandbox"))
    agent.add_node(sandbox)
    agent.add_edge(START, "call_model")
    agent.add_edge("sandbox", "call_model")
    return agent