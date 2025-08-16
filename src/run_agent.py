import os 
import sys
from llama_cpp import Llama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import tools

def format_prompt(instruction):
    """Formats the instruction for the model."""
    return f"""### Instruction:
{instruction}

### Response:
"""
#Main function to run our Astra Agent
def run_agent():
    model_path = "models/astra-coder-Q4_K_M.gguf"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    llm=Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )

    print("--- Astra Agent Initialized ---")
    print("Type your command or 'exit' to quit.")

    while True:
        instruction = input("\n> ")
        if instruction.lower() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break

    prompt = format_prompt(instruction)
    
    response = llm(
        prompt,
        max_tokens=256,
        stop=["\n"],
        echo=False
    )

    tool_call_str = response['choices'][0]['text'].strip()
    print(f"Astra says: `{tool_call_str}`")

    try:
        tool_name, args_str = tool_call_str.split("(", 1)
        args_str = args_str[:-1]
        args = [arg.strip().strip("'\"") for arg in args_str.split(',') if arg.strip()]

        if hasattr(tools, tool_name):
                func = getattr(tools, tool_name)
                result = func(*args)
                print("\n--- Tool Result ---")
                print(result)
                print("-------------------")
            else:
                print(f"Error: Tool '{tool_name}' not found in tools.py.")

        except Exception as e:
            print(f"Error: Could not execute the command. Details: {e}")