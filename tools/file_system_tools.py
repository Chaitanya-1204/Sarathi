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

    
