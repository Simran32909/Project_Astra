import json
import random

# --- Vocabulary for Generation ---

VERBS_READ = ["read", "show me", "display", "get the contents of", "print", "cat"]
VERBS_WRITE = ["write", "save", "output", "create a file", "put the text"]
VERBS_LINT = ["lint", "check", "analyze", "run pylint on", "review the code quality of"]
VERBS_GET_BODY = ["get the function body of", "extract the function", "show me the implementation of", "what does the function"]

FILE_PATHS = [
    "main.py", "src/utils.py", "README.md", "config.json", "api/v1/user_routes.py",
    "lib/auth.py", "scripts/data_processing.py", "docs/guide.md", "test.log",
    "db/queries.py", "models/user.py", "assets/style.css", "index.html",
    "docker-compose.yml", "requirements.txt", "src/database.py", "app.js"
]

FUNCTION_NAMES = [
    "calculate_sum", "parse_data", "process_request", "start_server", "get_user",
    "User", "login_user", "fetch_api_data", "run_validation", "save_to_db",
    "initialize_logger", "render_template", "handle_error", "connect_to_redis"
]

CONTENT_STRINGS = [
    "hello world", "import os", "{\"key\": \"value\"}", "# ASTRA GUIDE", "testing",
    "from flask import Flask", "def my_func():\\n    pass", "[1, 2, 3]", "Astra is an AI agent."
]

# --- Template Functions ---

def generate_read_file_sample():
    verb = random.choice(VERBS_READ)
    path = random.choice(FILE_PATHS)
    instruction = f"{verb} the file '{path}'"
    output = f"read_file('{path}')"
    return {"instruction": instruction, "output": output}

def generate_write_file_sample():
    verb = random.choice(VERBS_WRITE)
    path = random.choice(FILE_PATHS)
    content = random.choice(CONTENT_STRINGS)
    instruction = f"{verb} '{content}' to a file named '{path}'"
    output = f"write_file('{path}', '{content}')"
    return {"instruction": instruction, "output": output}

def generate_lint_code_sample():
    verb = random.choice(VERBS_LINT)
    path = random.choice(FILE_PATHS)
    instruction = f"{verb} '{path}'"
    output = f"lint_code('{path}')"
    return {"instruction": instruction, "output": output}

def generate_get_function_body_sample():
    verb = random.choice(VERBS_GET_BODY)
    func = random.choice(FUNCTION_NAMES)
    path = random.choice(FILE_PATHS)
    instruction = f"{verb} '{func}' from the file '{path}'"
    output = f"get_function_body('{path}', '{func}')"
    return {"instruction": instruction, "output": output}


# --- Main Generation Logic ---

def generate_dataset(num_samples):
    dataset = []
    generation_functions = [
        generate_read_file_sample,
        generate_write_file_sample,
        generate_lint_code_sample,
        generate_get_function_body_sample
    ]
    for _ in range(num_samples):
        # Choose a random function type to generate
        gen_func = random.choice(generation_functions)
        dataset.append(gen_func())
    return dataset

if __name__ == "__main__":
    NUM_TRAIN_SAMPLES = 500
    NUM_VAL_SAMPLES = 100 # A good rule of thumb is 10-20% of the training set

    print(f"Generating {NUM_TRAIN_SAMPLES} training samples...")
    train_dataset = generate_dataset(NUM_TRAIN_SAMPLES)
    with open("data/train_dataset.json", "w") as f:
        json.dump(train_dataset, f, indent=4)
    print("Training dataset saved to train_dataset.json")

    print(f"Generating {NUM_VAL_SAMPLES} validation samples...")
    val_dataset = generate_dataset(NUM_VAL_SAMPLES)
    with open("data/val_dataset.json", "w") as f:
        json.dump(val_dataset, f, indent=4)
    print("Validation dataset saved to val_dataset.json")
    
    print("\nDataset generation complete!")
