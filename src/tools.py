import os
import subprocess
import ast

#Read contents of a file
def read_file(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return str(e)

#Lints a Python file and returns the issues.
def lint_code(file_path):
    try:
        result = subprocess.run(['pylint', file_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

#Extracts the body of a function from a Python file.
def get_function_body(file_path, function_name):
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                body_nodes = node.body
                start_line = body_nodes[0].lineno
                end_line = body_nodes[-1].end_lineno
                
                lines = source.splitlines()
                return '\n'.join(lines[start_line-1:end_line])
        return f"Function '{function_name}' not found in '{file_path}'."
    except Exception as e:
        return str(e)

#Writes content to a file
def write_file(path, content):
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return str(e)