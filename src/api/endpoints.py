from typing import List, Union

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langserve import add_routes

from src.graph.workflow import create_graph


class ChatInputType(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]


# Load environment variables from .env file
load_dotenv()


def start() -> None:
    app = FastAPI(
        title="Memory Agent",
        version="0.1.0",
        description="A simple api server to test out my memory agent",
    )

    # Configure CORS
    origins = [
        "http://localhost",
        "http://localhost:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    graph = create_graph()

    # runnable = graph.with_types(input_type=ChatInputType, output_type=dict)

    add_routes(app, graph, path="/chat", config_keys=["configurable"], playground_type="default")

    print("Starting server...")
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    start()