from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from Curate import Curate
from pathlib import Path
import uvicorn
import httpx

# from fastapi import FastAPI
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles

# Mixin both FasttAPI and FastMCP, so we are able to define both traditional API and MCP endpoints.
# app = FastAPI()
mcp = FastMCP("Curator", port=8002)
dir_path = Path(__file__).parent
readme_file = dir_path / "readme.md"
static_path = dir_path / "static"

# This is a default FastAPI, we MOUNT the MCP server to it with the /mcp endpoint.
# app.mount("/mcp", mcp.sse_app())
# app.mount("/static", StaticFiles(directory=static_path), name="static")
#
#
# @app.get("/", response_class=HTMLResponse)
# def root():
#     return """
# <!DOCTYPE html>
# <html lang="en">
# <head>
# <title>Curate Server</title>
# <link rel="icon" href="/static/favicon.ico" type="image/x-icon">
# </head>
# <body>
# A simple server to run the Curate class.
# API and MCP endpoints are available.
# </body>
# </head>
# """
#

# MCP stuff specifically
## If you want to have a clean context manager for configs, db connections, etc., this is recommended best practice
# from contextlib import asynccontextmanager
# from collections.abc import AsyncIterator
# from pydantic import BaseModel
#
# class AppContext(BaseModel):
#     """


#     Application context for the FastMCP server.
#     This class can be used to store shared data or configuration.
#     """
#
#     pass
#
# @asynccontextmanager
# async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
#     """
#     Lifespan event handler for the FastMCP server.
#     This function is called when the server starts and stops.
#     Leaving blank for now, but this can be a source of configs, db connections, etc.
#     """
#     # Perform startup tasks here
#     print("Starting up the server...")
#
#     # Yield control to the server
#     yield AppContext()
#
#     # Perform shutdown tasks here
#     print("Shutting down the server...")
#
#
# mcp = FastMCP("Curator", lifespan=app_lifespan)


# Example resources
@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "App Configuration here"


@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Dynamic user data"""
    return f"Profile data for user {user_id}"


@mcp.resource("readme://")
def get_readme() -> str:
    """Readme file"""
    with open(readme_file, "r") as f:
        return f.read()


# Example tools
@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters"""
    return weight_kg / (height_m**2)


@mcp.tool()
async def fetch_weather(city: str) -> str:
    """Fetch current weather for a city."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text


@mcp.tool()
async def request_courses(query_string: str, k: int = 5, n_results: int = 30) -> str:
    """
    Curate function that uses the Curate class to process a query string.
    """
    results = Curate(query_string, k=k, n_results=n_results)
    output = ""
    for result in results:
        output += f"{result[0]}: {result[1]}\n"
    return output


# Example prompts
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far"),
    ]


if __name__ == "__main__":
    # Need to run this at uvicorn layer because of mounting
    # uvicorn.run(app, host="0.0.0.0", port=9000)
    mcp.run(transport="sse")
