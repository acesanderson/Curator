import pytest
import os
import pandas as pd
from unittest.mock import patch, mock_open
from io import StringIO
from datetime import datetime

# Import the functions from your main script
from Curate import (
	installed, cosmo_export_exists, vector_db_exists, date_manifest_exists,
	check_cosmo_export_last_modified, check_date_manifest, update_required,
	clean_text, load_cosmo_export, write_date_manifest, get_vector_db_collection,
	load_to_chroma, validate_chroma_database, create_vector_db, update_vector_db,
	query_vector_db, rerank_options, query_courses, process_multiline_input,
	process_input_file, batch_queries
)

# Mock data and configurations
MOCK_COSMO_FILE = "mock_courselist_en_US.xlsx"
MOCK_DATE_MANIFEST = ".mock_date_manifest"
MOCK_VECTOR_DB = ".mock_chroma_database"

@pytest.fixture
def mock_files(tmp_path):
	# Create mock files
	cosmo_path = tmp_path / MOCK_COSMO_FILE
	pd.DataFrame({
		'Course Name EN': ['Course 1', 'Course 2'],
		'Course Description': ['Desc 1', 'Desc 2'],
		'Activation Status': ['ACTIVE', 'ACTIVE'],
		'Course Release Date': ['2019-01-01', '2020-01-01'],
		'Course Updated Date': ['2019-01-01', '2020-01-01']
	}).to_excel(cosmo_path, index=False)
	
	date_manifest_path = tmp_path / MOCK_DATE_MANIFEST
	date_manifest_path.write_text("1609459200.0")  # 2021-01-01 00:00:00
	
	vector_db_path = tmp_path / MOCK_VECTOR_DB
	os.mkdir(vector_db_path)
	
	return {'cosmo': cosmo_path, 'date_manifest': date_manifest_path, 'vector_db': vector_db_path}


def test_installed(mock_files):
	with patch('Curate.cosmo_file', mock_files['cosmo']), \
		patch('Curate.date_manifest', mock_files['date_manifest']), \
		patch('Curate.vector_db', mock_files['vector_db']):
		assert installed() == True

def test_cosmo_export_exists(mock_files):
	with patch('Curate.cosmo_file', mock_files['cosmo']):
		assert cosmo_export_exists() == True

def test_vector_db_exists(mock_files):
	with patch('Curate.vector_db', mock_files['vector_db']):
		assert vector_db_exists() == True

def test_date_manifest_exists(mock_files):
	with patch('Curate.date_manifest', mock_files['date_manifest']):
		assert date_manifest_exists() == True

def test_check_cosmo_export_last_modified(mock_files):
	with patch('Curate.cosmo_file', mock_files['cosmo']):
		assert isinstance(check_cosmo_export_last_modified(), float)

def test_check_date_manifest(mock_files):
	with patch('Curate.date_manifest', mock_files['date_manifest']):
		assert check_date_manifest() == "1609459200.0"

def test_update_required(mock_files):
	with patch('Curate.cosmo_file', mock_files['cosmo']), \
		patch('Curate.date_manifest', mock_files['date_manifest']):
		assert isinstance(update_required(), bool)

def test_clean_text():
	assert clean_text("<p>Test &amp; text</p>") == "Test & text"

def test_load_cosmo_export(mock_files):
	with patch('Curate.cosmo_file', mock_files['cosmo']):
		data = load_cosmo_export()
		assert isinstance(data, list)
		assert len(data) == 2
		assert isinstance(data[0], tuple)

def test_write_date_manifest(mock_files, capsys):
	with patch('Curate.date_manifest', mock_files['date_manifest']):
		write_date_manifest("1609459200.0")
		captured = capsys.readouterr()
		assert "Date manifest created" in captured.out

@pytest.mark.parametrize("input_text,expected", [
	("line1\nline2", ["line1", "line2"]),
	("single line", ["single line"]),
])
def test_process_multiline_input(input_text, expected):
	assert process_multiline_input(input_text) == expected

def test_process_input_file(tmp_path):
	# Test CSV
	csv_file = tmp_path / "test.csv"
	csv_file.write_text("query1\nquery2")
	assert process_input_file(str(csv_file)) == ["query1", "query2"]
	
	# Test Excel
	excel_file = tmp_path / "test.xlsx"
	pd.DataFrame({"col1": ["query1", "query2"]}).to_excel(excel_file, index=False, header=False)
	assert process_input_file(str(excel_file)) == ["query1", "query2"]
	
	# Test TXT
	txt_file = tmp_path / "test.txt"
	txt_file.write_text("query1\nquery2")
	assert process_input_file(str(txt_file)) == ["query1", "query2"]

# More complex tests that might require mocking chromadb

@pytest.fixture
def mock_collection():
	class MockCollection:
		def query(self, query_texts, n_results):
			return {
				'ids': [['Course 1', 'Course 2']],
				'documents': [['Desc 1', 'Desc 2']]
			}
		
		def count(self):
			return 2
	return MockCollection()
