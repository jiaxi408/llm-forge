import inspect
from typing import Any

from langchain_sandbox import PyodideSandbox

from codeact import EvalCoroutine

def create_pyodide_eval_fn(sandbox: PyodideSandbox) -> EvalCoroutine:
    """Create an eval_fn that uses PyodideSandbox.
    """

    async def async_eval_fn(
        code: str, _locals: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:

        wrapper_code = f"""
def execute():
    try:
        # Execute the provided code
{chr(10).join("        " + line for line in code.strip().split(chr(10)))}
        return locals()
    except Exception as e:
        return {{"error": str(e)}}

execute()
"""
        # Convert functions in _locals to their string representation
        context_setup = ""
        for key, value in _locals.items():
            if callable(value):
                # Get the function's source code
                src = inspect.getsource(value)
                context_setup += f"\n{src}"
            else:
                context_setup += f"\n{key} = {repr(value)}"

        try:
            # Execute the code and get the result
            response = await sandbox.execute(
                code=context_setup + "\n\n" + wrapper_code,
            )

            # Check if execution was successful
            if response.stderr:
                return f"The error occurred when running your code:\n{response.stderr}\n", {}

            # Get the output from stdout
            output = (
                f"The code has been executed, and the output is as follows:\n{response.stdout}\n"
                if response.stdout
                else "The code has been executed with no output printed"
            )
            result = response.result

            # If there was an error in the result, return it
            if isinstance(result, dict) and "error" in result:
                return f"The error occurred when running your code:\n{result['error']}\n", {}

            # Get the new variables by comparing with original locals
            new_vars = {
                k: v
                for k, v in result.items()
                if k not in _locals and not k.startswith("_")
            }
            return output, new_vars

        except Exception as e:
            return f"Error during PyodideSandbox execution: {repr(e)}", {}

    return async_eval_fn