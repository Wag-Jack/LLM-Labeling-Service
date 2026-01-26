from huggingface_hub import snapshot_download
from pathlib import Path

curr_dir = Path.cwd()
DATA_DIR = curr_dir.parent / "Data/EdAcc"

# Run a snapshot download of EdAcc to Data directory
snapshot_download(
    repo_id="edinburghcstr/edacc",
    repo_type="dataset",
    local_dir=DATA_DIR,
    local_dir_use_symlinks=False,
)

print("Downloaded to:", DATA_DIR)