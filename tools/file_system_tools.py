import os
import shutil
from urllib.parse import urlparse
import git

from logger import log

CLONED_DIR = "./repositories"
def clone_repo(repo_url:str )-> str:
    
    """
    Clones a public Git repository to a local directory.

    If the repository has already been cloned, it returns the path to the existing
    directory without re-cloning.

    Args:
        repo_url: The URL of the public Git repository to clone.

    Returns:
        The local path to the cloned repository, or None if cloning fails.
    """
    
    
    # get the repo name from the url
    repo_name = repo_url.rstrip('/').split('/')[-1]
    if repo_name.endswith('.git'):
        repo_name = repo_name[:-4]
    
    # create the local path to clone the repo
    local_path = os.path.join(CLONED_DIR, repo_name)  
    
    
    # # check if repo already exist or not
    # if os.path.exists(local_path):
    #     log(f"Repository already exists at {local_path}.", prefix="Cloning")
    #     shutil.rmtree(local_path)  # Remove the existing directory
    #     log(f"Removed existing repository at {local_path}.", prefix="Cloning")
    
    
    #------------------To be removed in further just for testing ---------------------------
    
    
    if os.path.exists(local_path):
        log(f"Repository already exists at {local_path}.", prefix="Cloning")
        return local_path
    
    # ---------------------------------------------------------------------------------------
    
    # create the directory if it does not exist
    os.makedirs(CLONED_DIR, exist_ok=True)
    log(f"Cloning repository from {repo_url} to {local_path}.", prefix="Cloning")
    
    # clone the repository
    git.Repo.clone_from(repo_url , local_path )
    log(f"Repository cloned successfully to {local_path}.", prefix="Cloning")
    
    return local_path

    

def list_files_in_directory(relative_dir_path: str) -> list[str] | None:
    """
    Lists all files (not directories) in a given directory within the project's
    repository folder, with security checks to prevent directory traversal.

    Args:
        relative_dir_path (str): The path to the directory, relative to the
                                 main repository folder (e.g., "my-repo/src").

    Returns:
        A list of file names, or None if the path is invalid or an error occurs.
    """
    try:
        
        base_dir = os.path.abspath(CLONED_DIR)
    
        requested_path = os.path.abspath(os.path.join(base_dir, relative_dir_path))

        
        if not requested_path.startswith(base_dir):
            log(f"SECURITY ALERT: Directory traversal attempt blocked for path: {relative_dir_path}", prefix="ERROR")
            return None

        # Check if the path actually exists and is a directory.
        if not os.path.isdir(requested_path):
            log(f"Path is not a valid directory: {requested_path}", prefix="WARN")
            return None

        log(f"Listing files in: {requested_path}", prefix="File-List")

        # --- List only files, excluding subdirectories ---
        # os.listdir() gets all contents, os.path.isfile() filters for files.
        files_only = [
            f for f in os.listdir(requested_path) 
            if os.path.isfile(os.path.join(requested_path, f))
        ]
        
        return files_only

    except Exception as e:
        log(f"An unexpected error occurred while listing files in '{relative_dir_path}': {e}", prefix="ERROR")
        return None



def read_file_content(file_path: str) -> str:
    """
    Reads the content of a file, correctly handling and sanitizing paths from the agent.
    """
    try:
        # **FIX: Sanitize the input path to remove extra quotes or whitespace.**
        # This is the most important change to fix the error.
        sanitized_path = file_path.strip().strip("'\"`")

        # Get the absolute path for the allowed base directory
        allowed_base = os.path.abspath(CLONED_DIR)
        
        # Use the sanitized path for all subsequent operations
        requested_path = os.path.abspath(sanitized_path)

        # SECURITY CHECK: Ensure the resolved path is inside the allowed directory
        if not requested_path.startswith(allowed_base):
            print(f"SECURITY ALERT: Path '{requested_path}' is outside the allowed directory '{allowed_base}'.")
            return "Error: Access denied."

        with open(requested_path, "r", encoding="utf-8") as f:
            content = f.read()
            return content
            
    except FileNotFoundError:
        print(f"File not found for input: {sanitized_path}")
        return "Error: File not found."
    except Exception as e:
        print(f"An unexpected error occurred while reading {sanitized_path}: {e}")
        return f"An error occurred: {e}"