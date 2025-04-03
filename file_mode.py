# Curate is now split into server and file mode. This is file mode.
# This will be accessed by the main Curate script which is simplified.
# -----------------------------------------------------------------
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
import sys, os
import logging
from contextlib import contextmanager


# Tensorflow + pytorch have c++ libraries that are very verbose.
@contextmanager
def silence_all_output():
    """
    Context manager that completely silences both stdout and stderr by redirecting to /dev/null at the OS level.
    """
    # Save original file descriptors
    old_stdout = os.dup(1)
    old_stderr = os.dup(2)

    # Close and redirect stdout and stderr
    os.close(1)
    os.close(2)
    os.open(os.devnull, os.O_WRONLY | os.O_CREAT)
    os.dup2(1, 2)  # Redirect stderr to the same place as stdout

    try:
        yield
    finally:
        # Restore original stdout and stderr
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)


# Similarly, sentence transformers noisily reports all embedding creation, so we are setting global python logging to warnings only.
logging.basicConfig(level=logging.WARNING)

# our imports
# -----------------------------------------------------------------
console = Console(width=100)  # for spinner

with console.status("[green]Loading...", spinner="dots"):
    with silence_all_output():
        from Curator.rerank import rerank_options
        from Curator.cache.cache import CuratorCache, CachedQuery, CachedResponse
        import chromadb  # for vector database
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        import argparse  # for parsing command line arguments
        import pandas as pd  # for reading cosmo export + preparing data for vector database
        import os  # for getting date / time data from file
        import sys  # for sys.exit
        import html  # for cleaning text
        import re  # for cleaning text
        import shutil  # for deleting the vector database
        from datetime import datetime  # for reporting the day/time of the last update
        import time
        from pathlib import Path
        import torch
        import asyncio
        from asyncio import Semaphore
        from typing import Literal


# Configs
# -----------------------------------------------------------------

implementation: Literal["sync", "async"] = "sync"
model: Literal["gtr-t5-large", "all-MiniLM-L6-v2"] = "all-MiniLM-L6-v2"
device: Literal["cpu", "cuda", "mps", ""] = ""

# Detect device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Set SentenceTransformer's logger to only show warnings and above
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
ef = SentenceTransformerEmbeddingFunction(model_name=model, device=device)


# Definitions
# -----------------------------------------------------------------

script_dir = Path(__file__).resolve().parent
cosmo_file = str(
    script_dir / "courselist_en_US.xlsx"
)  # script needs three files to function; cosmo export, vector database, and date manifest
date_manifest = str(script_dir / ".date_manifest")
vector_db = str(script_dir / ".chroma_database")
checkbox = "[✓]"
# Set up the cache; set to None if you do not want caching.
cache = CuratorCache()
cache_path = script_dir / "cache" / ".curator_cache.db"

# Functions
# -----------------------------------------------------------------
## Application status checks -- booleans + data/time strings.


def installed(verbose=False) -> bool:
    """
    Check if everything is set up.
    """
    if verbose:
        with console.status("[green]Status...", spinner="dots"):
            time.sleep(1)
            console.print("\n")
            console.print("[green]Status[/green]")
            console.print(
                "[yellow]------------------------------------------------------------------------[/yellow]"
            )
            console.print(
                "COSMO EXPORT:                      " + str(cosmo_export_exists())
            )
            console.print(
                "VECTOR DB:                         " + str(vector_db_exists())
            )
            console.print(
                "DATE MANIFEST:                     " + str(date_manifest_exists())
            )
    checks = [cosmo_export_exists(), vector_db_exists(), date_manifest_exists()]
    if False in checks:
        return False
    else:
        return True


def cosmo_export_exists() -> bool:
    """
    Return True if the cosmo export file exists.
    """
    return os.path.exists(cosmo_file)


def vector_db_exists() -> bool:
    """
    Return True if the vector database exists.
    """
    return os.path.exists(vector_db)


def date_manifest_exists() -> bool:
    """
    Return True if the data manifest exists.
    """
    return os.path.exists(date_manifest)


def check_cosmo_export_last_modified() -> str:
    """
    Return the last modified time of the cosmo export file.
    """
    return os.path.getmtime(cosmo_file)


def check_date_manifest() -> str:
    """
    Return the str from the date_manifest file.
    """
    with open(date_manifest, "r") as f:
        last_updated = f.read()
    return last_updated


def update_required() -> bool:
    """
    Return True if the Cosmo export is newer than the last update.
    """
    last_modified = check_cosmo_export_last_modified()
    last_updated = float(check_date_manifest())
    if last_modified > last_updated:
        return True
    else:
        return False


## Handling the Cosmo Export
# -----------------------------------------------------------------


def clean_text(text):
    """
    This is useful for all Cosmo data you're bringing in through pandas.
    """
    # Decode HTML entities
    text = html.unescape(text)
    # Handle common encoding issues
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove any remaining HTML tags
    text = re.sub("<[^<]+?>", "", text)
    return text.strip()


def load_cosmo_export() -> pd.DataFrame:
    """
    Load the cosmo export file.
    """
    # Clear our cache
    if cache:
        cache.clear_cache()
    df = pd.read_excel(cosmo_file, engine="openpyxl")
    # Prepare the data
    df = df.fillna("")
    # Clean the text in both columns
    df["Course Name EN"] = df["Course Name EN"].apply(clean_text)
    df["Course Description"] = df["Course Description"].apply(clean_text)
    # Remove duplicates from both columns
    df = df.drop_duplicates(subset=["Course Name EN"])
    # Filter for rows that are marked "ACTIVE" in the "Activation Status" column
    df = df[df["Activation Status"] == "ACTIVE"]
    # Filter it for rows that have a "Course Release Date" after 1/1/2018 OR have a value in "Course Updated Date" after 1/1/2018.
    df = df[
        (df["Course Release Date"] > "2018-01-01")
        | (df["Course Updated Date"] > "2018-01-01")
    ]
    return df  # type: ignore


def get_data() -> list[tuple]:
    """
    Get the data from the cosmo export file.
    Wrapper that calls load_cosmo_export.
    """
    df = load_cosmo_export()
    # Get a list of tuples, first item is Course Name EN, second item is Course Description
    data = list(zip(df["Course Name EN"], df["Course Description"]))
    return data


def write_date_manifest(last_updated: str) -> None:
    """
    Writes the last updated date to the date manifest file.
    """
    with open(date_manifest, "w") as f:
        f.write(last_updated)
    console.print(f"[green]{checkbox} Date manifest created: {date_manifest}[/green]")


def print_readme() -> None:
    """
    Simple function to print the readme as a help file.
    """
    from rich.markdown import Markdown

    with open("readme.md", "r") as f:
        markdown_text = f.read()
    console.print(Markdown(markdown_text))


## Handling the Vector Database
# -----------------------------------------------------------------


def get_vector_db_collection() -> chromadb.Collection:
    """
    Get the vector database collection.
    """
    client = chromadb.PersistentClient(vector_db)
    collection = client.get_collection(name="descriptions", embedding_function=ef)
    return collection


def update_progress(current, total) -> None:
    """
    This takes the index and len(iter) of a for loop and creates a pretty
    progress bar.
    """
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    if current != total:
        percent = float(current) * 100 / total
        bar = (
            GREEN
            + "=" * int(percent)
            + RESET
            + YELLOW
            + "-" * (100 - int(percent))
            + RESET
        )
        print(
            f"\rProgress: |{bar}| {current} of {total} | {percent:.2f}% Complete",
            end="",
        )
        sys.stdout.flush()
    elif current == total:
        print("\rProgress: |" + "=" * 100 + f"| {current} of {total} | 100% Complete\n")


def load_to_chroma(collection: chromadb.Collection, data: list[tuple[str, float]]):
    """
    Load the descriptions into the chroma database.
    """
    for index, datum in enumerate(data):
        course_title, description = datum
        if index % 10 == 0:
            update_progress(index, len(data))
        collection.add(
            ids=[course_title],
            documents=[description],
        )
    print("Data loaded to chroma database.")


async def _load_to_chroma_async(
    collection: chromadb.Collection, chunk: list, sem: Semaphore
):
    pass


async def load_to_chroma_async(
    collection: chromadb.Collection, data: list[tuple[str, float]], sem: Semaphore
):
    """
    Load the descriptions into the chroma database asynchronously.
    """

    # Batch items into chunks of 100; enumerate them so we can pass the index to the coroutine
    async def load_batch(collection, chunk, sem):
        async with sem:
            ids = [datum[0] for datum in chunk]
            documents = [datum[1] for datum in chunk]
            await collection.add(
                ids=ids,
                documents=documents,
            )
            update_progress(len(chunk), len(data))

    chunks = list(enumerate([data[i : i + 100] for i in range(0, len(data), 100)]))
    print(f"Loading {len(chunks)} docs into Chroma...")
    coroutines = [load_batch(collection, chunk, sem) for chunk in chunks]
    await asyncio.gather(*coroutines)


def validate_chroma_database(collection: chromadb.Collection) -> None:
    """
    Validate the chroma database.
    """
    # Time of last update
    timestamp_str = check_date_manifest()
    timestamp = float(timestamp_str)
    dt_object = datetime.fromtimestamp(timestamp)
    readable_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    print("Last updated:               ", readable_time)
    # Number of docs
    print("Number of courses:                ", collection.count())


def create_vector_db() -> chromadb.Collection:
    """
    Create the vector databases.
    """
    console.print(
        "[green]Generating embeddings for the course descriptions. This may take a while.[/green]"
    )
    # Load cosmo export
    data = get_data()
    if implementation == "sync":
        # Delete existing database
        if os.path.exists(vector_db):
            shutil.rmtree(vector_db)
        # Create the new database
        client = chromadb.PersistentClient(path=vector_db)
        # Create the collection
        collection = client.create_collection(
            name="descriptions", embedding_function=ef
        )
        # Add the data to the collection
        load_to_chroma(collection, data)
    # elif implementation == "async":
    #     # Create a semaphore to limit the number of concurrent requests
    #     sem = Semaphore(10)
    #     # Load our client and collection
    #     client = await chromadb.AsyncHttpClient(port=8001)
    #     try:
    #         await client.delete_collection(name="Curator")
    #     except:
    #         pass
    #     collection = await client.create_collection(
    #         name="Curator", embedding_function=ef
    #     )
    #     load_to_chroma_async(collection, data, sem)
    # Write the date manifest
    print("Writing date manifest.")
    write_date_manifest(str(check_cosmo_export_last_modified()))
    validate_chroma_database(collection)
    return collection


def update_vector_db() -> chromadb.Collection:
    """
    Update the vector database.
    """
    print("Loading new Cosmo export and generating embeddings for the new courses.")
    # Load the existing collection
    collection = get_vector_db_collection()
    # Load the cosmo export
    data = load_cosmo_export()
    # Identifying the new data
    all_courses = set([datum[0] for datum in data])
    processed_courses = set(collection.get()["ids"])
    new_courses = all_courses - processed_courses
    new_data = [datum for datum in data if datum[0] in new_courses]
    # Add the data to the collection
    load_to_chroma(collection, new_data)
    print(f"Added {len(new_data)} new courses to the database.")
    validate_chroma_database(collection)
    # Write the date manifest
    print("Writing date manifest.")
    write_date_manifest(str(check_cosmo_export_last_modified()))
    return collection


def get_all_ids() -> list[int]:
    """
    Get all the ids from the collection.
    """


## Our query functions
# -----------------------------------------------------------------


def query_vector_db(
    collection: chromadb.Collection, query_string: str, n_results: int
) -> list[tuple]:
    """
    Query the collection for a query string and return the top n results.
    """
    results = collection.query(query_texts=[query_string], n_results=n_results)
    ids = results["ids"][0]
    documents = results["documents"][0]
    responses = list(zip(ids, documents))
    return responses


def query_courses(
    collection: chromadb.Collection,
    query_string: str,
    k: int = 5,
    n_results: int = 30,
    model_name: str = "bge",
) -> list[tuple]:
    """
    Query the collection for a query string and return the top n results.
    """
    query_string = query_string.strip()
    console.print(
        "[yellow]------------------------------------------------------------------------[/yellow]"
    )
    # Check if the query is in the cache
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
        results = query_vector_db(collection, query_string, n_results)
        # Rerank the results
        reranked_results = rerank_options(results, query_string, k, model_name)
    # Add to the cache
    if cache:
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
    query_string: str, k: int = 5, n_results: int = 30, model_name: str = "mxbai"
) -> list[tuple]:
    """
    This is the importable version of the query_courses function.
    Returns a list of tuples, where the first element is the course title and the second is the confidence.
    """
    if installed():
        if update_required():
            print("New data found. Updating our vector database...")
            collection = update_vector_db()
        else:
            collection = get_vector_db_collection()
    results = query_courses(
        collection=collection,
        query_string=query_string,
        k=k,
        n_results=n_results,
        model_name=model_name,
    )
    return results


## Batch processing
# -----------------------------------------------------------------


def process_multiline_input(text: str) -> list[str]:
    """
    Process a multiline text.
    """
    return text.split("\n")


def process_input_file(filename: str) -> list[str]:
    """
    Process an input file -- assumption is that the data is in the first column.
    """
    if filename.endswith(".csv"):
        # note: there is no column title in the csv, so treat first cell as data
        df = pd.read_csv(filename, header=None)
        return df.iloc[:, 0].tolist()
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(filename, header=None)
        return df.iloc[:, 0].tolist()
    elif filename.endswith(".txt"):
        with open(filename, "r") as f:
            data = f.readlines()
        return [line.strip() for line in data]


def batch_queries(queries: list[str], k, n) -> list[list[tuple[str, float]]]:
    """
    Wrapper query_courses for multiple queries.
    """
    console.print("\n")
    console.print(
        f"Processing {len(queries)} queries. Press ctrl-c at any time to exit."
    )
    batch_results = []
    for index, query in enumerate(queries):
        results = query_courses(collection, query, k=k, n_results=n)
        batch_results.append(results)
        console.print(
            f"[green]Query {index + 1} of {len(queries)}:[/green] [yellow]{query}[/yellow]"
        )
        console.print(
            "[yellow]------------------------------------------------------------------------[/yellow]"
        )
        for result in results:
            console.print(result)
    return batch_results


# Output file
# -----------------------------------------------------------------


def create_hyperlink(url: str, course_title: str) -> str:
    """
    Generate Excel / Google Sheets-friendly hyperlink.
    """
    # url can have multiple urls separated by a comma, grab the first
    url = url.split(",")[0]
    return f'=HYPERLINK("{url}", "{course_title}")'


def load_cosmo_metadata() -> pd.DataFrame:
    """
    Load cosmo data for our output dataframe.
    """
    # Load cosmo file as a dataframe
    df = pd.read_excel(cosmo_file)
    # Prepare the data
    df = df.fillna("")
    # Clean the text in both columns
    df["Course Name EN"] = df["Course Name EN"].apply(clean_text)
    df["Course Description"] = df["Course Description"].apply(clean_text)
    # Remove duplicates from both columns
    df = df.drop_duplicates(subset=["Course Name EN"])
    # Filter for rows that are marked "ACTIVE" in the "Activation Status" column
    df = df[df["Activation Status"] == "ACTIVE"]
    # Filter it for rows that have a "Course Release Date" after 1/1/2018 OR have a value in "Course Updated Date" after 1/1/2018.
    df = df[
        (df["Course Release Date"] > "2018-01-01")
        | (df["Course Updated Date"] > "2018-01-01")
    ]
    # First, create a new column called Course Link. This value is created from the two existing columns in df: create_hyperlink(LIL URL, Course Title EN)
    df["Course Link"] = df.apply(
        lambda x: create_hyperlink(x["LIL URL"], x["Course Name EN"]), axis=1
    )
    # We want a new df that is a slice of original df, these columns: Course ID, Course Name EN, LI Level EN, Manager Level, Internal Library, Internal Subject, Course Duration, and Course Link
    cosmo_df = df[
        [
            "Course ID",
            "Course Name EN",
            "Course Link",
            "Course Description",
            "Course Release Date",
            "Course Updated Date",
            "LI Level EN",
            "Manager Level",
            "Internal Library",
            "Internal Subject",
            "Visible Duration",
        ]
    ]
    return cosmo_df


def create_output_dataframe(
    query: str, results: list[tuple[str, float]], cosmo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create a dataframe from the results.
    """
    # Convert results object (which is a list of tuples, first element is Course Name EN and second is Confidence) into a dataframe called results_df
    results_df = pd.DataFrame(results, columns=["Course Name EN", "Confidence"])
    # Add a column titled "Query" which has the query string in every row.
    results_df["Query"] = query
    # Query should be the first column
    results_df = results_df[["Query", "Course Name EN", "Confidence"]]
    # Now, we want to add the data from output_df to results_df, if Course Name EN matches Course Name EN in output_df. Add all columns from output_df except Course Name EN.
    results_df = results_df.merge(cosmo_df, on="Course Name EN", how="left")
    return results_df


def create_output_dataframe_batch(
    queries: list, results: list[list[tuple]], cosmo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Bulk wrapper for create_output_dataframe.
    """
    # Validate that lists are same length.
    if len(queries) != len(results):
        print("Queries and results are not the same length.")
        return None
    bulk_df = pd.DataFrame()
    for index, query in enumerate(queries):
        results_df = create_output_dataframe(query, results[index], cosmo_df)
        bulk_df = pd.concat([bulk_df, results_df])
    return bulk_df


if __name__ == "__main__":
    # Check if everything is installed
    if installed():
        if update_required():
            print("New data found. Updating our vector database...")
            collection = update_vector_db()
        else:
            collection = get_vector_db_collection()
    else:
        if not cosmo_export_exists():
            console.print(
                "[red]Cosmo export not found. Please download the latest export from Cosmo and try again.[/red]"
            )
            sys.exit()
        else:
            # First time! Installation with a little bit of fanfare.
            # Clear the terminal
            if os.name == "nt":
                os.system("cls")
            else:
                os.system("clear")
            text = Text(
                "     Welcome to Curator: context-driven course recommendations      ",
                style="bold white",
            )
            welcome_card = Panel(
                text,
                title=None,
                expand=False,
                border_style="bold",
                padding=(1, 1),
                width=90,
            )
            console.print(welcome_card)
            console.print("[green]First time installation:[/green]")
            console.print(f"[green]Device detected: {device}[/green]")
            console.print(f"[green]{checkbox} Cosmo export found: {cosmo_file}[/green]")
            collection = create_vector_db()
            # Delete the db file at cache_path
            if os.path.exists(cache_path):
                os.remove(cache_path)
                console.print(
                    f"[green]{checkbox} Cache file refreshed: {cache_path}[/green]"
                )
            console.print(
                f"[green]{checkbox} Vector database created: {vector_db}[/green]"
            )
            console.print(
                "[italic yellow]First-time user? Run the script with `python Curate.py -r` to see the readme.[/italic yellow]"
            )
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
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        help="Input filename (either csv or txt or excel; data needs to be in a single column)",
    )
    parser.add_argument("-o", "--output_file", type=str, help="Output filename")
    parser.add_argument("-r", "--readme", action="store_true", help="Print the readme.")
    # parser.add_argument('-w', '--wipe', type=str, help='Delete data files to start fresh.')
    args = parser.parse_args()
    status = args.status
    query = args.query
    if args.readme:
        print_readme()
        sys.exit()
    if status:
        installed(True)
        validate_chroma_database(collection)
        sys.exit()
    if args.number_responses:
        k = args.number_responses
    else:
        k = 5
    if args.original_batch_size:
        n = args.original_batch_size
    else:
        n = 50
    if args.input_file:
        queries = process_input_file(args.input_file)
        results = batch_queries(queries, k, n)
        if args.output_file:  # We'll use create_output_dataframe_batch here
            with console.status(
                "[green]Creating output and writing to CSV...", spinner="dots"
            ):
                cosmo_df = load_cosmo_metadata()
                bulk_df = create_output_dataframe_batch(queries, results, cosmo_df)
                bulk_df.to_csv(args.output_file + ".csv", index=False)
            console.print(
                f"\n[yellow]Results written to file: {args.output_file + '.csv'}[/yellow]"
            )
        sys.exit()
    elif query:
        if "\n" in query:
            queries = process_multiline_input(query)
            results = batch_queries(queries, k, n)
            if args.output_file:  # We'll use create_output_dataframe_batch here
                with console.status(
                    "[green]Creating output and writing to CSV...", spinner="dots"
                ):
                    cosmo_df = load_cosmo_metadata()
                    bulk_df = create_output_dataframe_batch(queries, results, cosmo_df)
                    bulk_df.to_csv(args.output_file + ".csv", index=False)
                console.print(
                    f"\n[yellow]Results written to file: {args.output_file + '.csv'}[/yellow]"
                )
            sys.exit()
        results = query_courses(collection, query, k=k, n_results=n)
        console.print(f"[green]Query: {query}[/green]")
        console.print(
            "[yellow]------------------------------------------------------------------------[/yellow]"
        )
        for result in results:
            print(result)
        if args.output_file:  # We'll use create_output_dataframe here
            with console.status(
                "[green]Creating output and writing to CSV...", spinner="dots"
            ):
                with open(args.output_file + ".csv", "w") as f:
                    cosmo_df = load_cosmo_metadata()
                    output_df = create_output_dataframe(query, results, cosmo_df)
                    output_df.to_csv(f, index=False)
                console.print(
                    f"\n[yellow]Results written to file: {args.output_file + '.csv'}[/yellow]"
                )
