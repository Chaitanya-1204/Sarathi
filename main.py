from tools.file_system_tools import clone_repo
from logger import log

def preprocess(repo_url):
    
    cloned_path = clone_repo(repo_url)
    
    log(f"Preprocessing completed. Cloned repository is available at: {cloned_path}", prefix="Preprocessing")
    
    return cloned_path
    


if __name__ == "__main__":
    
    
    repo_url = "https://github.com/salesforce/BLIP.git"
    preprocess(repo_url)
    