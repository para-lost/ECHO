# download script
from huggingface_hub import snapshot_download

snapshot_download(
    "fal/AuraFace-v1",
    local_dir="./models/auraface", # put the path you want to save the model
)