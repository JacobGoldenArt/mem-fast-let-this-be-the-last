import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from src.graph.workflow import workflow

load_dotenv()
# Create a graph from the workflow
graph = workflow.compile()


async def main():
    inputs = {"messages": [HumanMessage(content=input("Enter a message: "))]}
    # Run the graph
    while True:
        async for output in graph.astream(inputs, stream_mode="updates"):
            # stream_mode="updates" yields dictionaries with output keyed by node name
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                # Debugging: Print the entire output to understand its structure
                print(value)
                # Check if 'messages' key exists in the value dictionary
                if "messages" in value:
                    # Handle AIMessage object correctly
                    if isinstance(value["messages"], list):
                        print(value["messages"][-1].pretty_print())
                    else:
                        print(value["messages"].content)
                else:
                    print("No 'messages' key found in the output.")
            print("\n---\n")
        if inputs["messages"][0].content.lower() == "exit":
            print("Goodbye!")
            break
        inputs = {"messages": [HumanMessage(content=input("Enter a message: "))]}
        print("\n---\n")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())