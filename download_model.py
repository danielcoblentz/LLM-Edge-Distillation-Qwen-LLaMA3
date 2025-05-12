from huggingface_hub import snapshot_download

'''


This script downloads specified models from Hugging Face Hub only run if you want to download models again.

'''


HF_TOKEN = ""  # Replace with  Hugging Face token after login
def main():
    models = {
        "Qwen-7B": {
            "repo_id": "Qwen/Qwen2.5-7B-Instruct",
            "local_dir": "models/qwen-7b"
        },
        "Llama-3-8B": {
            "repo_id": "meta-llama/Meta-Llama-3-8B",
            "local_dir": "models/llama-3-8b"
        },
    }

    for model_name, details in models.items():
        print(f"Downloading {model_name}...")
        snapshot_download(repo_id=details["repo_id"], local_dir=details["local_dir"], token=HF_TOKEN)
        print(f"Download completed: {model_name}")

    print("All models have been downloaded successfully.")


if __name__ == "__main__":
    main()