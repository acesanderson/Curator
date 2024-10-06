# Imports can take a while, so we'll give the user a spinner.
# -----------------------------------------------------------------
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time

console = Console(width=100) # for spinner

# our imports
# -----------------------------------------------------------------

with console.status("[bold green]Loading...", spinner="dots"):
	# time.sleep(1)
	import chromadb         # for vector database
	import argparse         # for parsing command line arguments
	import pandas as pd     # for reading cosmo export + preparing data for vector database
	import os               # for getting date / time data from file
	import sys              # for sys.exit
	import html				# for cleaning text
	import re				# for cleaning text
	import shutil			# for deleting the vector database
	from datetime import datetime	# for reporting the day/time of the last update
	from FlagEmbedding import FlagReranker	# for reranking
	import time
	from pathlib import Path

# Definitions
# -----------------------------------------------------------------

script_dir = Path(__file__).resolve().parent
cosmo_file = str(script_dir / "courselist_en_US.xlsx") # script needs three files to function; cosmo export, vector database, and date manifest
date_manifest = str(script_dir / ".date_manifest")
vector_db = str(script_dir / ".chroma_database")
checkbox = "[✓]"

# Functions
# -----------------------------------------------------------------
## Application status checks -- booleans + data/time strings.

def installed(verbose = False) -> bool:
	"""
	Check if everything is set up.
	"""
	if verbose:
		with console.status("[bold green]Status...", spinner="dots"):
			time.sleep(1)
			console.print("\n")
			console.print("[green]Status[/green]")
			console.print("[yellow]------------------------------------------------------------------------[/yellow]")
			console.print("COSMO EXPORT:                      " + str(cosmo_export_exists()))
			console.print("VECTOR DB:                         " + str(vector_db_exists()))
			console.print("DATE MANIFEST:                     " + str(date_manifest_exists()))
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
	with open(date_manifest, 'r') as f:
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
	text = text.encode('ascii', 'ignore').decode('ascii')
	# Remove any remaining HTML tags
	text = re.sub('<[^<]+?>', '', text)
	return text.strip()

def load_cosmo_export() -> list[tuple]:
	"""
	Load the cosmo export file.
	"""
	df = pd.read_excel(cosmo_file)
	# Prepare the data
	df = df.fillna('')
	# Clean the text in both columns
	df['Course Name EN'] = df['Course Name EN'].apply(clean_text)
	df['Course Description'] = df['Course Description'].apply(clean_text)
	# Remove duplicates from both columns
	df = df.drop_duplicates(subset=['Course Name EN'])
	# Filter for rows that are marked "ACTIVE" in the "Activation Status" column
	df = df[df['Activation Status'] == 'ACTIVE']
	# Filter it for rows that have a "Course Release Date" after 1/1/2018 OR have a value in "Course Updated Date" after 1/1/2018.
	df = df[(df['Course Release Date'] > '2018-01-01') | (df['Course Updated Date'] > '2018-01-01')]
	# Get a list of tuples, first item is Course Name EN, second item is Course Description
	data = list(zip(df['Course Name EN'], df['Course Description']))
	return data

def write_date_manifest(last_updated: str) -> None:
	"""
	Writes the last updated date to the date manifest file.
	"""
	with open(date_manifest, 'w') as f:
		f.write(last_updated)
	console.print(f"[green]{checkbox} Date manifest created: {date_manifest}[/green]")

def print_readme() -> None:
	"""
	Simple function to print the readme as a help file.
	"""
	from rich.markdown import Markdown
	with open('readme.md', 'r') as f:
		markdown_text = f.read()
	console.print(Markdown(markdown_text))

## Handling the Vector Database
# -----------------------------------------------------------------

def get_vector_db_collection() -> chromadb.Collection:
	"""
	Get the vector database collection.
	"""
	client = chromadb.PersistentClient(vector_db)
	collection = client.get_collection(name="descriptions")
	return collection

def update_progress(current, total) -> None:
	"""
	This takes the index and len(iter) of a for loop and creates a pretty
	progress bar.
	"""
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RESET = '\033[0m'
	if current != total:
		percent = float(current) * 100 / total
		bar = GREEN + "=" * int(percent) + RESET + YELLOW + '-' * (100 - int(percent)) + RESET
		print(f'\rProgress: |{bar}| {current} of {total} | {percent:.2f}% Complete', end='')
		sys.stdout.flush()
	elif current == total:
		print('\rProgress: |' + '=' * 100 + f'| {current} of {total} | 100% Complete\n')

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
	console.print("[green]Generating embeddings for the course descriptions. This may take a while.[/green]")
	# Delete existing database
	if os.path.exists(vector_db):
		shutil.rmtree(vector_db)
	# Create the new database
	client = chromadb.PersistentClient(path=vector_db)
	# Create the collection
	collection = client.create_collection(name="descriptions")
	# Load cosmo export
	data = load_cosmo_export()
	# Add the data to the collection
	load_to_chroma(collection, data)
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
	processed_courses = set(collection.get()['ids'])
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

def load_reranker() -> None:
	"""
	Load the reranker; this can take a while when first initializing.
	"""
	with console.status(f'[bold green]Installing reranking model... This may take a while. [/bold green]', spinner="dots"):
		reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

## Our query functions
# -----------------------------------------------------------------

def query_vector_db(collection: chromadb.Collection, query_string: str, n_results: int) -> list[str]:
	"""
	Query the collection for a query string and return the top n results.
	"""
	results = collection.query(
		query_texts=[query_string],
		n_results=n_results
	)
	ids = results['ids'][0]
	documents = results['documents'][0]
	return list(zip(ids, documents))

def rerank_options(options: list[tuple], query: str, k: int = 5) -> list[tuple]:
	"""
	Reranking magic.
	"""
	reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
	ranked_results: list[tuple] = []
	for option in options:
		course = option[0]				# This is "id" from the Chroma output.
		TOC = option[1]					# This is "document" from the Chroma output.
		score = reranker.compute_score([query, TOC])
		ranked_results.append((course, score))
	# sort ranked_results by highest score
	ranked_results.sort(key=lambda x: x[1], reverse=True)
	# Return the five best.
	return ranked_results[:k]

def query_courses(collection: chromadb.Collection, query_string: str, k: int = 5, n_results: int = 30) -> list[tuple]:
	"""
	Query the collection for a query string and return the top n results.
	"""
	console.print("[yellow]------------------------------------------------------------------------[/yellow]")
	with console.status(f'[bold green]Query: [/bold green][yellow]"{query_string}"[/yellow][green]...[/green]', spinner="dots"):
		time.sleep(1)
		results = query_vector_db(collection, query_string, n_results)
		reranked_results = rerank_options(results, query_string, k)
	return reranked_results

def Curate(query_string: str, k: int = 5, n_results: int = 30) -> list[tuple]:
	"""
	This is the importable version of the query_courses function.
	"""
	if installed():
		if update_required():
			print("New data found. Updating our vector database...")
			collection = update_vector_db()
		else:
			collection = get_vector_db_collection()
	results = query_courses(collection = collection, query_string = query_string, k = k, n_results = n_results)
	return results

## Batch processing
# -----------------------------------------------------------------

def process_multiline_input(text: str) -> str:
	"""
	Process a multiline text.
	"""
	return text.split('\n')

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
		with open(filename, 'r') as f:
			data = f.readlines()
		return [line.strip() for line in data]

def batch_queries(queries: list[str], k, n) -> list[list[tuple[str, float]]]:
	"""
	Wrapper query_courses for multiple queries.
	"""
	console.print('\n')
	console.print(f"Processing {len(queries)} queries. Press ctrl-c at any time to exit.")
	batch_results = []
	for index, query in enumerate(queries):
		results = query_courses(collection, query, k = k, n_results = n)
		batch_results.append(results)
		console.print(f"[green]Query {index + 1} of {len(queries)}:[/green] [yellow]{query}[/yellow]")
		console.print("[yellow]------------------------------------------------------------------------[/yellow]")
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
	url = url.split(',')[0]
	return f'=HYPERLINK("{url}", "{course_title}")'

def load_cosmo_metadata() -> pd.DataFrame:
	"""
	Load cosmo data for our output dataframe.
	"""
	# Load cosmo file as a dataframe
	df = pd.read_excel(cosmo_file)
	# Prepare the data
	df = df.fillna('')
	# Clean the text in both columns
	df['Course Name EN'] = df['Course Name EN'].apply(clean_text)
	df['Course Description'] = df['Course Description'].apply(clean_text)
	# Remove duplicates from both columns
	df = df.drop_duplicates(subset=['Course Name EN'])
	# Filter for rows that are marked "ACTIVE" in the "Activation Status" column
	df = df[df['Activation Status'] == 'ACTIVE']
	# Filter it for rows that have a "Course Release Date" after 1/1/2018 OR have a value in "Course Updated Date" after 1/1/2018.
	df = df[(df['Course Release Date'] > '2018-01-01') | (df['Course Updated Date'] > '2018-01-01')]
	# First, create a new column called Course Link. This value is created from the two existing columns in df: create_hyperlink(LIL URL, Course Title EN)
	df['Course Link'] = df.apply(lambda x: create_hyperlink(x['LIL URL'], x['Course Name EN']), axis=1)
	# We want a new df that is a slice of original df, these columns: Course ID, Course Name EN, LI Level EN, Manager Level, Internal Library, Internal Subject, Course Duration, and Course Link
	cosmo_df = df[['Course ID', 'Course Name EN', 'Course Link', 'Course Release Date', 'Course Updated Date', 'LI Level EN', 'Manager Level', 'Internal Library', 'Internal Subject', 'Visible Duration']]
	return cosmo_df

def create_output_dataframe(query: str, results: list[tuple[str, float]], cosmo_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Create a dataframe from the results.
	"""
	# Convert results object (which is a list of tuples, first element is Course Name EN and second is Confidence) into a dataframe called results_df
	results_df = pd.DataFrame(results, columns=['Course Name EN', 'Confidence'])
	# Add a column titled "Query" which has the query string in every row.
	results_df['Query'] = query
	# Query should be the first column
	results_df = results_df[['Query', 'Course Name EN', 'Confidence']]
	# Now, we want to add the data from output_df to results_df, if Course Name EN matches Course Name EN in output_df. Add all columns from output_df except Course Name EN.
	results_df = results_df.merge(cosmo_df, on='Course Name EN', how='left')
	return results_df

def create_output_dataframe_batch(results: list[list[tuple]]) -> pd.DataFrame:
	"""
	Bulk wrapper for create_output_dataframe.
	"""
	pass

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
			console.print("[red]Cosmo export not found. Please download the latest export from Cosmo and try again.[/red]")
			sys.exit()
		else:
			# First time! Installation with a little bit of fanfare.
			# Clear the terminal
			os.system('clear')
			text = Text("     Welcome to Curator: context-driven course recommendations      ", style="bold white")
			welcome_card = Panel(
				text,
				title=None,
				expand=False,
				border_style="bold",
				padding=(1, 1),
				width=90
			)
			console.print(welcome_card)
			console.print("[bold green]First time installation:[/bold green]")
			console.print(f"[green]{checkbox} Cosmo export found: {cosmo_file}[/green]")
			load_reranker()			
			console.print(f"[green]{checkbox} Reranker installed.[/green]")
			collection = create_vector_db()
			console.print(f"[green]{checkbox} Vector database created: {vector_db}[/green]")
			console.print("[italic yellow]First-time user? Run the script with `python Curate.py -r` to see the readme.[/italic yellow]")
	# Our arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('query', nargs='?', help='A query for the text.')
	parser.add_argument('-n', '--original_batch_size', type=int, help='Number of responses: this is 50 by default.')
	parser.add_argument('-k', '--number_responses', type=int, help='Original pool size: this is 5 by default.')
	parser.add_argument('-s', '--status', action="store_true", help="Print the status of the application")
	parser.add_argument('-i', '--input_file', type=str, help='Input filename (either csv or txt or excel; data needs to be in a single column)')
	parser.add_argument('-o', '--output_file', type=str, help='Output filename')
	parser.add_argument('-r', '--readme', action="store_true", help='Print the readme.')
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
		if args.output_file:
			with open(args.output_file, 'w') as f:
				for result in results:
					f.write(str(result) + '\n')
			console.print(f"\n[yellow]Results written to file: {args.output_file}[/yellow]")
		sys.exit()
	elif query:
		if '\n' in query:
			queries = process_multiline_input(query)
			print(queries)
			results = batch_queries(queries, k, n)
			if args.output_file: # We'll use create_output_dataframe_batch here
				with open(args.output_file, 'w') as f:
					for result in results:
						f.write(str(result) + '\n')
				console.print(f"\n[yellow]Results written to file: {args.output_file}[/yellow]")
			sys.exit()
		results = query_courses(collection, query, k = k, n_results = n)
		console.print(f"[green]Query: {query}[/green]")
		console.print("[yellow]------------------------------------------------------------------------[/yellow]")
		for result in results:
			print(result)
		if args.output_file: # We'll use create_output_dataframe here
			with console.status("[bold green]Creating output and writing to CSV...", spinner="dots"):
				with open(args.output_file + '.csv', 'w') as f:
					cosmo_df = load_cosmo_metadata()
					output_df = create_output_dataframe(query, results, cosmo_df)
					output_df.to_csv(f, index=False)
			console.print(f"\n[yellow]Results written to file: {args.output_file + '.csv'}[/yellow]")
