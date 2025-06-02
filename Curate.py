from Kramer import (
    query_course_descriptions_sync,
    query_course_descriptions,
)  # The latter is async
from Curator.rerank import rerank_options, rerank_options_async
from Curator.cache.cache import CuratorCache, CachedQuery, CachedResponse
import argparse  # for parsing command line arguments
import time
from pathlib import Path
from rich.console import Console


# Set up the cache; set to None if you do not want caching.
dir_path = Path(__file__).resolve().parent
db_path = dir_path / "cache" / ".curator_cache.db"
cache = CuratorCache(db_path=db_path)
# Our console
console = Console(width=100)  # for spinner


## Our query functions
# -----------------------------------------------------------------


def query_courses(
    query_string: str,
    k: int = 5,
    n_results: int = 30,
    model_name: str = "bge",
    cached=True,
) -> list[tuple]:
    """
    Query the collection for a query string and return the top n results.
    This is a wrapper for the query_course_descriptions function from Kramer.
    """
    query_string = query_string.strip()
    console.print(
        "[yellow]------------------------------------------------------------------------[/yellow]"
    )
    # Check if the query is in the cache
    if cached:
        cache_hit = cache.cache_lookup(query_string.lower())
        if cache_hit:
            console.print(f"[green]Query found in cache: {query_string}[/green]")
            cache_hit = [
                (response.course_title, response.similarity) for response in cache_hit
            ]
            return cache_hit
    # Otherwise, query the vector database
    with console.status(
        f'[green]Query: [/green][yellow]"{query_string}"[/yellow][green]...[/green]',
        spinner="dots",
    ):
        time.sleep(1)
        # Get the results from the vector database
        results = query_course_descriptions_sync(query_string, n_results)
        # Rerank the results
        reranked_results = rerank_options(results, query_string, k, model_name)
    # Add to the cache
    if cache and cached:
        cached_responses = [
            CachedResponse(
                course_title=reranked_result[0], similarity=reranked_result[1]
            )
            for reranked_result in reranked_results
        ]
        cached_query = CachedQuery(
            query=query_string.lower(), responses=cached_responses
        )
        cache.insert_cached_query(cached_query)
    return reranked_results


async def query_courses_async(
    query_string: str,
    k: int = 5,
    n_results: int = 30,
    model_name: str = "bge",
    cached=True,
) -> list[tuple]:
    """
    Query the collection for a query string and return the top n results.
    """
    query_string = query_string.strip()
    console.print(
        "[yellow]------------------------------------------------------------------------[/yellow]"
    )
    # Check if the query is in the cache
    if cached:
        cache_hit = cache.cache_lookup(query_string.lower())
        if cache_hit:
            console.print(f"[green]Query found in cache: {query_string}[/green]")
            cache_hit = [
                (response.course_title, response.similarity) for response in cache_hit
            ]
            return cache_hit
        # Otherwise, query the vector database
    with console.status(
        f'[green]Query: [/green][yellow]"{query_string}"[/yellow][green]...[/green]',
        spinner="dots",
    ):
        time.sleep(1)
        # Get the results from the vector database
        results = await query_course_descriptions(query_string, n_results)
        # Rerank the results
        reranked_results = await rerank_options_async(
            results, query_string, k, model_name
        )
    # Add to the cache
    if cache and cached:  # If cache exists and cached is chosed
        cached_responses = [
            CachedResponse(
                course_title=reranked_result[0], similarity=reranked_result[1]
            )
            for reranked_result in reranked_results
        ]
        cached_query = CachedQuery(
            query=query_string.lower(), responses=cached_responses
        )
        cache.insert_cached_query(cached_query)
    return reranked_results


def Curate(
    query_string: str,
    k: int = 5,
    n_results: int = 30,
    model_name: str = "bge",
    cached=True,
) -> list[tuple]:
    """
    This is the importable version of the query_courses function.
    Returns a list of tuples, where the first element is the course title and the second is the confidence.
    """
    results = query_courses(
        query_string=query_string,
        k=k,
        n_results=n_results,
        model_name=model_name,
        cached=cached,
    )
    return results


async def CurateAsync(
    query_string: str,
    k: int = 5,
    n_results: int = 30,
    model_name: str = "bge",
    cached=True,
) -> list[tuple]:
    """
    This is the async version of the Curate function.
    """
    results = await query_courses_async(
        query_string=query_string,
        k=k,
        n_results=n_results,
        model_name=model_name,
        cached=cached,
    )
    return results


if __name__ == "__main__":
    # Our arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="A query for the text.")
    parser.add_argument(
        "-n",
        "--original_batch_size",
        type=int,
        help="Number of responses: this is 50 by default.",
    )
    parser.add_argument(
        "-k",
        "--number_responses",
        type=int,
        help="Original pool size: this is 5 by default.",
    )
    parser.add_argument(
        "-s",
        "--status",
        action="store_true",
        help="Print the status of the application",
    )
    args = parser.parse_args()
    status = args.status
    query = args.query
    if args.number_responses:
        k = args.number_responses
    else:
        k = 5
    if args.original_batch_size:
        n = args.original_batch_size
    else:
        n = 50
    results = query_courses(query, k=k, n_results=n)
    console.print(f"[green]Query: {query}[/green]")
    console.print(
        "[yellow]------------------------------------------------------------------------[/yellow]"
    )
    for result in results:
        print(result)
