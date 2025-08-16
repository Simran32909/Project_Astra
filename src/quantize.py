import os 
import hydra
from omegaconf import DictConfig
import subprocess

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def quantize(cfg:DictConfig):
    merged_model_path = os.path.join(cfg.trainer.output_dir, "merged_model")

    llama_cpp_path = "vendor/llama.cpp" 
    
    convert_script = os.path.join(llama_cpp_path, "convert.py")
    quantize_executable = os.path.join(llama_cpp_path, "build/bin/quantize")

    if not os.path.exists(quantize_executable) or not os.path.exists(convert_script):
        raise FileNotFoundError("llama.cpp tools not found. Looks like the submodule is not built correctly.")
    
    if not os.path.exists(merged_model_path):
        raise FileNotFoundError(f"Merged model not found at {merged_model_path}")

    gguf_output_dir = "Astra/models" 
    f16_gguf_name = "astra-coder-f16.gguf"
    f16_gguf_path = os.path.join(gguf_output_dir, f16_gguf_name)

    os.makedirs(gguf_output_dir, exist_ok=True)

    print(f"Converting model to GGUF at {f16_gguf_path}...")
    convert_cmd = [
        "python", convert_script, merged_model_path,
        "--outfile", f16_gguf_path,
        "--outtype", "f16"
    ]
    subprocess.run(convert_cmd, check=True)
    print(f"Model converted to GGUF at {f16_gguf_path}")

    quantized_gguf_name = "astra-coder-Q4_K_M.gguf"
    quantized_gguf_path = os.path.join(gguf_output_dir, quantized_gguf_name)

    print(f"Quantizing model to {quantized_gguf_path}...")
    quantize_cmd = [
        quantize_executable, f16_gguf_path, quantized_gguf_path, "Q4_K_M"
    ]
    subprocess.run(quantize_cmd, check=True)
    print(f"Model quantized successfully.")

if __name__ == "__main__":
    quantize()