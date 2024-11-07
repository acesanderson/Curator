# Use an official Python runtime as a parent image.
FROM python:3.12.4-slim

# Set the working directory, in the containerr
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/
COPY courselist_en_US.xlsx /app/
COPY input.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application codde into the container
COPY . /app

# Command to run the script
CMD ["python", "Curate.py", "-i", "input.txt", "-o", "output"]


