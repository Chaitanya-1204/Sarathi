
from logger import log
from tools.file_system_tools import clone_repo
from tools.code_search_tool import create_vector_store



        
def preprocess(repo_url):
    
    # clone the repository
    cloned_path = clone_repo(repo_url)
    log(f"Preprocessing completed. Cloned repository is available at: {cloned_path}", prefix="Preprocessing")
    
    # Create the vector store
    vector_store = create_vector_store(cloned_path)
    log(f"Vector store created successfully ", prefix="Preprocessing")
    
    
    return vector_store