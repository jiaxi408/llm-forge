import re

BACKTICK_PATTERN = r"(?:^|\n)```(.*?)(?:```(?:\n|$))"

def extract_and_combine_codeblocks(text: str) -> str:
    code_blocks = re.findall(BACKTICK_PATTERN, text, re.DOTALL)

    if not code_blocks:
        return ""

    processed_block = []
    for block in code_blocks:
        lines = block.split("\n")
        block = "\n".join(lines[1:-1])
        processed_block.append(block)

    combined_code = "\n\n".join(processed_block)
    return combined_code