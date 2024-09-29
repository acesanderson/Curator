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

# Definitions

cosmo_file = "courselist_en_US.xlsx"
date_manifest = ".date_manifest"
vector_db = ".chroma_database"

# Functions

## Application status checks -- booleans + data/time strings.

def installed(verbose = False) -> bool:
	"""
	Check if everything is set up.
	"""
	if verbose:
		print("COSMO EXPORT:                      " + str(cosmo_export_exists()))
		print("VECTOR DB:                         " + str(vector_db_exists()))
		print("DATE MANIFEST:                     " + str(date_manifest_exists()))
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
	return os.path.exists(".date_manifest")

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
	print("Date manifest updated.")

## Handling the Vector Database

def get_vector_db_collection() -> chromadb.Collection:
	"""
	Get the vector database collection.
	"""
	client = chromadb.PersistentClient(path=vector_db)
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

def load_to_chroma(collection: chromadb.Collection, data: list[tuple[str, str]]):
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
	print("Loading Cosmo export and generating embeddings. This may take a while.")
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
	validate_chroma_database(collection)
	# Write the date manifest
	print("Writing date manifest.")
	write_date_manifest(str(check_cosmo_export_last_modified()))
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

## Our query functions

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

def query_courses(collection: chromadb.Collection, query_string: str, k: int, n_results: int) -> list[str]:
	"""
	Query the collection for a query string and return the top n results.
	"""
	results = query_vector_db(collection, query_string, n_results)
	reranked_results = rerank_options(results, query_string, k)
	return reranked_results

if __name__ == "__main__":
	# Check if everything is installed
	if installed():
		print("======================\nCurator 1.0.\n======================")
		if update_required():
			print("New data found. Updating our vector database...")
			collection = update_vector_db()
		else:
			collection = get_vector_db_collection()
	else:
		print("Initializing Curator...")
		if not cosmo_export_exists():
			print("Cosmo export not found. Please download the latest export from Cosmo and try again.")
			sys.exit()
		else:
			collection = create_vector_db()
	# Our arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('query', nargs='?', help='A query for the text.')
	parser.add_argument('-n', '--original_batch_size', type=int, help='Number of responses: this is 50 by default.')
	parser.add_argument('-k', '--number_responses', type=int, help='Original pool size: this is 5 by default.')
	parser.add_argument('-s', '--status', action="store_true", help="Print the status of the application")
	# parser.add_argument('-i', '--input_file', type=str, help='Input filename (either csv or txt or excel; data needs to be in a single column)')
	# parser.add_argument('-o', '--output_file', type=str, help='Output filename')
	# parser.add_argument('-r', '--readme', action="store_true", help="Print the README")
	args = parser.parse_args()
	status = args.status
	query = args.query
	if status:
		installed(True)
		sys.exit()
	if args.number_responses:
		k = args.number_responses
	else:
		k = 5
	if args.original_batch_size:
		n = args.original_batch_size
	else:
		n = 50
	if query:
		results = query_courses(collection, query, k = k, n_results = n)
		for result in results:
			print(result)







