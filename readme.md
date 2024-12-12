# Curator: Context-Driven Course Recommendations

## Overview

Curator is a command-line tool designed to provide context-driven course recommendations based on user queries. It uses advanced natural language processing techniques to match user inputs with relevant courses from a Cosmo export.

## Software Design

### Command Line Usage

Basic usage:
```
curate "your course query here"
```

This returns the top 5 most relevant courses by default.

### Arguments

- First positional argument (str): User query (optional)
- `-i`, `--input_file` (str): Input filename (CSV, TXT, or Excel; data should be in the first column)
- `-o`, `--output_file` (str): Output filename for results
- `-n`, `--original_batch_size` (int): Number of initial results to retrieve (default: 50)
- `-k`, `--number_responses` (int): Number of final recommendations to return (default: 5)
- `-s`, `--status`: Print the status of the application
- `-r`, `--readme`: Print this readme

## How It Works

![Two stage ranking model](two-stage-pipeline.png "Two stage ranking model")
*from [rerankers GitHub repo](https://www.answer.ai/posts/2024-09-16-rerankers.html)*

1. **Data Source**: The script uses a Cosmo export (Excel file) containing course titles and descriptions.

2. **Data Preprocessing**:
   - Filters for active courses
   - Includes courses released or updated after January 1, 2018
   - Cleans text data (removes HTML, handles encoding issues)

3. **Vector Database**: 
   - Creates embeddings of course descriptions using ChromaDB
   - Stores these embeddings in a persistent vector database for quick similarity searches

4. **Query Processing**:
   - Converts user queries into embeddings
   - Performs a similarity search against the course description embeddings

5. **Reranking**:
   - Uses a locally-hosted LLM reranking model (BAAI/bge-reranker-large) to improve recommendation quality
   - Reranks the initial results based on relevance to the query

6. **Output**: 
   - Returns the top k results (default k=5, can be modified)

Note: This script uses only locally-hosted code, ensuring data security.

## Installation for users new to Python (and command line)

**Setting up on Windows:**

1.  Download and install Python version 3.12.4 from their official website ([here](https://www.python.org/downloads/windows/))
- Be sure to check the box “Add Python 3.x to PATH” at the beginning of installation      

2.  Once installed, verify the installation
    1.  Open Command Prompt (you can search for “cmd” in the Start menu) 
    2.  Type into Command Prompt: **python –version**
    3.  You should then see the Python version you are currently running, it will need to be least version 3.7
        
3.  Download and install Git
    1.  Download and install Git for Windows ([here](https://git-scm.com/downloads/win))
    2.  During installation, keep the default setting which will add Git to your PATH
        
4.  Create a folder on your Desktop and call it something like “Curator” or “Curation Script”. For this documentation I will imagine you use “Curator”

5.  Clone the repository
    1.  Open the Command Prompt and navigate to the folder where you want to place your project (the one you created in Step 4). 
        1.  Use the following command: **cd %USERPROFILE%\\Desktop\\Curator**
    2.  Clone the repository
        1.  Type: **git clone https://github.com/acesanderson/Curator**
    3.  Change to the directory of the repository
        1.  **cd Curator**
            
6.  Install the requirements
    1.  Type: pip install -r requirements.txt
        
7.  Download the Cosmo export file and place it in the same folder you created in Step 4

8.  Run the script
    1.  Type: **python Curate.py “insert keywords here”**
    2.  When you run this for the first time, you will have to wait several minutes for the vector database to be created. This will happen automatically and you will be alerted when it is set up.

**Setting up on MacOs:**

1.  From the GitHub repo click the green “Code” dropdown and download the Zip file
    1.  Go to Finder on your Mac, go to your Downloads, unzip the folder, move it to your Desktop, and rename it “Curator” (the default name will be “Curator-main”)

2.  Go to [go/cosmoexports] and download the file called courselist\_en\_US.csv
    1.  Go to Finder and move this file from your Downloads to your newly created “Curator” folder
    2.  Once in the Curator folder, duplicate this document, and save the duplicate as a .xlsx file. Keep this duplicate in the same folder. 

3.  Open up Terminal. A quick way to do this on Mac is to press Cmd+Space and search for Terminal

4.   Once open, copy and paste the following line into the Terminal and press Enter. 
    1.  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    2.  You may be prompted to input your password. This will be the same password you use to log in to your device. Type the password and press Enter. Then press Enter again when prompted. 
        
5.  Copy and paste the following command into your Terminal and press Enter. This will install Python. 
    1.  brew install python
        
6.  Once complete, verify the installation using the following command. The number that appears needs to be at least version 3.7
    1.  python3 --version
        
7.  Install the repo using the following command. 
    1.  git clone https://github.com/acesanderson/Curator
        
8.  Go to the repo using the following command. 
    1.  cd Curator
        
9.  Install the necessary package requirements using the following command. 
    1.  pip install -r requirements.txt
        
10.  Start the curator script using the following command. 
    1.  python3 Curate.py
        
11.  You may receive an alert to download openpyxl or to upgrade pip. If this occurs, type these commands, depending on the alert. 
    1.  pip install openpyxl
    2.  python3 -m pip install --upgrade pip
        
12.  At this point, the script will create the vector database. This may take as long as 10 minutes to complete. You can monitor the progress in the terminal. Once complete, you will be alerted, and you will be able to run your queries. 
    
13.  To run a basic query use the following command.
    1.  python3 Curate.py “input query here”

## How to Use

==Depending on your system, you may need to use `python3` instead of `python` in the below example.==

1. **First-time setup**:
   - Run the script. It will automatically create the vector database on first run.
   ```
   python curate.py
   ```

2. **Basic query**:
   ```
   python curate.py "your course query here"
   ```

3. **Batch processing from file**:
   ```
   python curate.py -i input_file.csv -o results.txt
   ```

4. **Adjusting result count**:
   ```
   python curate.py "query" -k 10
   ```

5. **Check application status**:
   ```
   python curate.py -s
   ```

## As Python Module

The Curate function can also be imported and used as a Python module. Here's an example:

```
>>> import Curator
>>> from Curator import Curate
>>> c=Curate("machine learning with javascript")
------------------------------------------------------------------------
>>> c
[('Learning TensorFlow with JavaScript', 3.30859375), ('Level Up: JavaScript', 0.61474609375), ('JavaScript: Functions', 0.2330322265625), ('AI Programming for JavaScript Developers', -0.044281005859375), ('Hands-On Introduction: JavaScript', -0.32568359375)]
```

You will either need to:
- add to $PYTHONPATH the path to the directory containing the Curator.py file, or
- install the Curator package using pip:

```bash
pip install . # from the directory containing Curate.py.
```

Then you can import and use the Curate function as shown above.

## Best Practices

- Provide detailed queries for better results. The script matches your query to courses based on semantic similarity to their descriptions.
- While course titles can work as queries, you'll get better results by providing more context.
- For batch processing, ensure your input file has one query per line or in the first column.
- You can change two parameters: the number of initial results to retrieve (`-n`) and the number of final recommendations to return (`-k`). The default values are 50 and 5, respectively. Why change the number of initial results? The more results you retrieve, the more accurate the final recommendations might be, though potentially at the cost of longer processing times.
 
## Maintenance

- The script automatically checks if the Cosmo export has been updated and refreshes the vector database accordingly.
- To force a complete rebuild of the vector database, delete the `.chroma_database` directory and `.date_manifest` file, then run the script again.

## Troubleshooting

- If you encounter any issues, first ensure all dependencies are correctly installed and that the Cosmo export file is present and up-to-date.
- Check the application status using the `-s` flag for diagnostics.

For further assistance or to report issues, please contact Brian Anderson at bianderson@linkedin.com.
