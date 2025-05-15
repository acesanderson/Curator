from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from Curate import CurateAsync
from pathlib import Path

mcp = FastMCP("Curator")
dir_path = Path(__file__).parent
readme_file = dir_path / "readme.md"
static_path = dir_path / "static"


# Example resources
@mcp.resource("readme://")
def get_readme() -> str:
    """Readme file"""
    with open(readme_file, "r") as f:
        return f.read()


# Example tools
@mcp.tool()
async def request_courses(query_string: str, k: int = 5, n_results: int = 30) -> str:
    """
    Search for and retrieve course information based on a query string.

       Parameters:
           query_string: The search query to find relevant courses
           k: Number of top matching courses to consider initially (vector db search)
           n_results: Maximum number of results to return in the final output (reranker model)
    """
    results = await CurateAsync(query_string, k=k, n_results=n_results, cached=True)
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
    mcp.run()
