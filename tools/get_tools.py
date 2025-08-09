from langchain.tools import Tool
from functools import partial

from tools.file_system_tools import clone_repo , list_files_in_directory , read_file_content 
from tools.code_search_tool import search_codebase 


def build_tools(vector_store):
    """
    This function is used to build the tools for the application.
    
    """
    
    search_tool_func = partial(search_codebase , vector_store=vector_store)
    
    # 
    
    # Assuming your tool functions are defined
    tools = [
        Tool(
            name="Code_Search",
            func=search_tool_func,
            description="Use this tool FIRST to find relevant code or file paths by searching the entire codebase."
        ),
        Tool(
            name="Read_File_Content",
            func=read_file_content,
            description="Use this tool to read the entire content of a single file once you have its full path from 'Code_Search'."
        ),
        Tool(
            name="List_Directory_Contents",
            func=list_files_in_directory,
            description="Use this tool to list files in a directory. Input MUST be a directory path."
        )
    ]

    
    return tools
    