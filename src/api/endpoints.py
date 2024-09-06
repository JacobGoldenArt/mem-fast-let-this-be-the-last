import logging
from typing import List, Union

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import HumanMessage
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langserve import add_routes
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from starlette.responses import StreamingResponse

from src.graph.workflow import create_graph

# Set up Rich console
console = Console()

# Set up logging
# Configure logging to use Rich
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


class ChatInputType(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ..., description="The messages to be sent to the chatbot."
    )
    temperature: float = Field(
        default=0.3,
        description="The temperature to use for the chatbot.",
    )

    class Config:
        arbitrary_types_allowed = True


def start() -> None:
    app = FastAPI(
        title="Memory Agent",
        version="0.1.0",
        description="A simple api server to test out my memory agent",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "https://emmm.local/"],
        allow_methods=["*"],
        allow_headers=["Content-Type", "Accept"],
        allow_credentials=True,
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        console.rule("[bold blue]Request Information")

        # Request details
        console.print(f"[bold cyan]Request:[/bold cyan] {request.method} {request.url}")

        # Body
        body = await request.body()
        rprint(body)

        # Process the request
        response = await call_next(request)

        # Check if the response is a streaming response
        if isinstance(response, StreamingResponse):
            console.print(f"[bold yellow]Streaming response detected[/bold yellow]")

            # Collect streamed data
            async def stream_collector(response_generator):
                async for chunk in response_generator:
                    console.print(f"[bold yellow]Stream chunk:[/bold yellow] {chunk}")
                    yield chunk

            response = StreamingResponse(
                stream_collector(response.body_iterator), media_type=response.media_type
            )

        # Response status
        console.print(
            f"[bold yellow]Response status:[/bold yellow] {response.status_code}"
        )
        console.print(
            f"[bold yellow]Response headers:[/bold yellow] {response.headers}"
        )

        console.rule("[bold blue]End of Request")
        console.print()  # Add an empty line for separation

        return response

    graph = create_graph()

    add_routes(
        app,
        graph.with_types(input_type=ChatInputType, output_type=dict),
        path="/chat",
        config_keys=["configurable"],
    )

    print("Starting server...")
    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    start()
