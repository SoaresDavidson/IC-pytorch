import kagglehub

# Download latest version
path = kagglehub.dataset_download("sovitrath/road-pothole-images-for-pothole-detection")

print("Path to dataset files:", path)