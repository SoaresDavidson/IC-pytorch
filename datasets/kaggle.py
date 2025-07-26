import kagglehub

# Download latest version
path = kagglehub.dataset_download("vitaliykinakh/stable-imagenet1k")

print("Path to dataset files:", path)