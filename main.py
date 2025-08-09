
import os
from dotenv import load_dotenv

load_dotenv()

from tools.file_system_tools import clone_repo ,read_file_content , list_files_in_directory
from logger import log
from tools.code_search_tool import create_vector_store
from tools.get_tools import build_tools
from utils import create_test_questions, test_agent_and_save_results

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def preprocess(repo_url):
    
    # clone the repository
    cloned_path = clone_repo(repo_url)
    log(f"Preprocessing completed. Cloned repository is available at: {cloned_path}", prefix="Preprocessing")
    
    # Create the vector store
    vector_store = create_vector_store(cloned_path)
    log(f"Vector store created successfully ", prefix="Preprocessing")
    
    
    return vector_store
    


if __name__ == "__main__":
    
    
    repo_url = "https://github.com/salesforce/BLIP.git"
    vector_store = preprocess(repo_url)
    
    tools = build_tools(vector_store)
    
    api_key = os.getenv("OPENAI_API_KEY")
    

    # Pass the API key directly to the client for robust handling
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    
    
   

    
    prompt_template = """  
    
        You are a coding assistant specialized in navigating and answering questions about a large codebase. You have access to the following tools:

        {tools}

        **Your Strategy and Rules:**
        1.  **Discover and Inspect:** Always start by using `Code_Search` to locate relevant files. Once a promising file is found, immediately use `Read_File_Content` to inspect its source code.
        2.  **Synthesize First:** After you have read a file's content, your immediate next step is to analyze it and try to formulate the final answer. **Do not** search for the definitions of every helper function unless you are absolutely unable to answer the question with the information you already have.
        3.  **Self-Correction:** If you find yourself repeating the same action without getting new, useful results, you are stuck. You **must** change your approach and try a different tool or a different query.

        Use the following format strictly for your responses:
        \n

        Question: the input question you must answer\n
        Thought: you should always think about what to do\n
        Action: the action to take, should be one of [{tool_names}]\n
        Action Input: the input to the action\n
        Observation: the result of the action\n
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer\n
        Final Answer: the final answer to the original input question and it should be detailed\n
        \n
        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
        
        
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
        
    )
    log("Agent created successfully.", prefix="Initialization")

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    log("Agent executor created successfully.", prefix="Initialization")
    
    
    
    test_questions = create_test_questions()
    
    test_agent_and_save_results(
        agent=agent_executor,
        questions=test_questions[:3],
        output_file="agent_test_results.md"
    )   