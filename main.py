from agent_system import ReActAgent
from language_model import hf_llm
from knowledge_base import TOOLS

def main():
    print("Initializing ReAct Agent...")
    # Initialize the agent with the LLM and the tools
    agent = ReActAgent(llm=hf_llm, tools=TOOLS)

    # Define a query based on the knowledge base (Nestle 2024 report)
    query = "What were Nestle's total sales in 2024 and how much profit did they earn?"
    
    print(f"Running query: {query}")
    print("-" * 50)
    
    # Run the agent
    result = agent.run(query)
    
    print("-" * 50)
    print("Final Answer:", result.get("final_answer"))
    print("-" * 50)

if __name__ == "__main__":
    main()
