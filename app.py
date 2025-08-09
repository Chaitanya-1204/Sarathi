import os
import threading
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv


from utils import preprocess
from tools.get_tools import build_tools
from logger import log


from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage


# Load environment variables from .env file
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)



agent_state = {
    "vector_store": None,
    "agent_executor": None,
    "is_ready": False,
    "error": None
}


PROMPT_TEMPLATE = """
You are a coding assistant. Your goal is to answer questions about a codebase using the tools provided.

**TOOLS:**
You have access to the following tools:
{tools}

**CHAT HISTORY:**
Use the following conversation history for context:
{chat_history}

**INSTRUCTIONS & FORMAT:**
You absolutely MUST follow this format. Do not write any other text.

1.  **Thought:** Briefly think about your next step.
2.  **Action:** Choose ONE tool from this list: [{tool_names}].
3.  **Action Input:** Specify the input for the chosen tool.
4.  **Observation:** This is the result of the action.

...(You can repeat the Thought/Action/Action Input/Observation block multiple times)...


**Thought:** I now have enough information to provide the final answer.

When you have gathered enough information to answer the user's question, you MUST output the final answer in the following format:

**Final Answer:** [Your detailed answer, written in simple English, between 300 and 500 words]

**IMPORTANT:** Your entire response must end with the "Final Answer:" block. Do not add any text or thoughts after the final answer.

Begin!

**Question:** {input}
**Thought:**{agent_scratchpad}
"""

def initialize_agent(vector_store):
    """
    Initializes the LangChain agent and executor.
    """
    try:
        log("Building tools...", prefix="AgentInit")
        tools = build_tools(vector_store)

        log("Initializing LLM...", prefix="AgentInit")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key
        )

        log("Creating prompt template...", prefix="AgentInit")
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        log("Creating ReAct agent...", prefix="AgentInit")
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        log("Creating Agent Executor...", prefix="AgentInit")
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms to the format.",
            max_iterations=10, # Added to prevent infinite loops
        )

        log("Agent initialized successfully.", prefix="AgentInit")
        return agent_executor
    except Exception as e:
        log(f"Error during agent initialization: {e}", prefix="ERROR")
        raise
    
    
@app.route('/')
def index():
    """
    Renders the main HTML page.
    """
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def handle_preprocess():
    """
    Handles the GitHub URL submission, clones the repo, and creates the vector store.
    This is a long-running task, so it's handled here.
    """
    global agent_state
    repo_url = request.json.get('repo_url')

    if not repo_url:
        return jsonify({"error": "GitHub URL is required."}), 400

    # Reset state
    agent_state = {"vector_store": None, "agent_executor": None, "is_ready": False, "error": None}

    try:
        log(f"Starting preprocessing for {repo_url}", prefix="Preprocess")
        vector_store = preprocess(repo_url)
        agent_executor = initialize_agent(vector_store)

        agent_state["vector_store"] = vector_store
        agent_state["agent_executor"] = agent_executor
        agent_state["is_ready"] = True
        log("Preprocessing and agent initialization complete.", prefix="Preprocess")

        return jsonify({"message": "Analysis complete. You can now start chatting."})

    except Exception as e:
        log(f"An error occurred during preprocessing: {e}", prefix="ERROR")
        agent_state["error"] = str(e)
        return jsonify({"error": f"Failed to analyze the repository. {e}"}), 500


@app.route('/chat', methods=['POST'])
def handle_chat():
    """
    Handles incoming chat messages from the user.
    """
    global agent_state
    if not agent_state.get("is_ready") or not agent_state.get("agent_executor"):
        return jsonify({"error": "Agent is not ready. Please analyze a repository first."}), 400

    data = request.json
    user_message = data.get('message')
    chat_history_json = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Message is required."}), 400

    # Reconstruct chat history for the agent
    chat_history = []
    for msg in chat_history_json:
        if msg.get('role') == 'user':
            chat_history.append(HumanMessage(content=msg.get('content')))
        elif msg.get('role') == 'ai':
            chat_history.append(AIMessage(content=msg.get('content')))

    try:
        log(f"Invoking agent with message: {user_message}", prefix="Chat")
        response = agent_state["agent_executor"].invoke({
            "input": user_message,
            "chat_history": chat_history
        })

        ai_response = response.get('output', "I couldn't find an answer.")
        log(f"Agent response: {ai_response}", prefix="Chat")

        return jsonify({"answer": ai_response})

    except Exception as e:
        log(f"An error occurred during chat: {e}", prefix="ERROR")
        return jsonify({"error": "An internal error occurred while processing your message."}), 500


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs("repositories", exist_ok=True)
    os.makedirs("vector_stores", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    app.run(debug=True, port=5001)
