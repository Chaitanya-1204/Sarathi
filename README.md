# Sarathi
Sarathi - The AI Codebase Companion


# Project-Description 

Sarathi is an intelligent AI agent designed to dramatically accelerate the onboarding process for developers joining a new project. It acts as an interactive, conversational guide to any codebase. By cloning a repository and creating a searchable knowledge base, Sarathi allows new hires to ask complex questions in natural language—from "Where is the authentication logic handled?" to "Explain the purpose of the use-custom-hook.js file"—and receive instant, context-aware answers. The agent is powered by a state-of-the-art language model (GPT-4o-mini) and equipped with a suite of tools to read files, navigate the directory structure, and perform semantic searches across the entire codebase, effectively serving as an ever-present senior developer for new team members.

The core of the system is a React-based chatbot interface that provides a user-friendly way to interact with the backend agent. This agent first ingests a target GitHub repository, breaking down the code into manageable chunks, generating vector embeddings, and storing them in a vector database. When a user asks a query, the agent uses the vector database to find the most relevant code sections and files, then uses its file-system tools and LLM reasoning capabilities to synthesize a comprehensive and accurate answer.