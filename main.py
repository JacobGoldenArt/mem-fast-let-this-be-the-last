# main.py
import asyncio

from langchain_core.messages import HumanMessage

from src.graph.workflow import memgraph
from src.settings.config import config_manager


async def main():
    # Allow for command-line arguments or user input to set these values
    chosen_model = input("Enter model name (press Enter for default): ") or None
    chosen_temperature = input("Enter temperature (press Enter for default): ") or None

    # Create initial configuration with potential overrides
    initial_config = config_manager.create_config(
        model=chosen_model if chosen_model else None,
        temperature=float(chosen_temperature) if chosen_temperature else None
    )

    inputs = {"messages": [HumanMessage(content=input("Enter a message: "))]}

    while True:
        async for output in memgraph.astream(inputs, initial_config, stream_mode="updates"):
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                print(value)
                if "messages" in value:
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
    asyncio.run(main())