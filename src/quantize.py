import os 
import hydra
from omegaconf import DictConfig
import subprocess

@hydra.main(config_path="../configs", config_name="config.yaml")
def quantize(cfg:DictConfig):
    merged_model_path=os.path.join(cfg.trainer.output_dir, "merged_model")

    llama_cpp_path = os.getenv("LLAMA_CPP_PATH")
    if not llama_cpp_path or not os.path.exists(llama_cpp_path):
        print("LLAMA_CPP_PATH is not set or the path does not exist")
        return
    
    if not os.path.exists(merged_model_path):
        raise FileNotFoundError(f"Merged model not found at {merged_model_path}")
    
    convert_script = os.path.join(llama_cpp_path, "convert.py")
    gguf_output_dir="../models"
    f16_gguf_name="astra-coder-f16.gguf"
    f16_gguf_path=os.path.join(gguf_output_dir, f16_gguf_name)

    print(f"Converting model to GGUF at {f16_gguf_path}...")
    convert_cmd = [
        "python", convert_script, merged_model_path,
        "--outfile", f16_gguf_path,
        "--outtype", "f16"
    ]

    subprocess.run(convert_cmd, check=True)
    print(f"Model converted to GGUF at {f16_gguf_path}")

    #Here Quantization is defined
    quantize_script = os.path.join(llama_cpp_path, "quantize")
    quantized_gguf_name = "astra-coder-Q4_K_M.gguf"
    quantized_gguf_path = os.path.join(gguf_output_dir, quantized_gguf_name)

    print(f"Quantizing model to {quantized_gguf_path}...")

    quantize_cmd=[
        quantize_script, final_gguf_path, quantized_gguf_path, "Q4_K_M"
    ]
    subprocess.run(quantize_command, check=True)
    print(f"Model quantized successfully.")

if __name__ == "__main__":
    quantize()