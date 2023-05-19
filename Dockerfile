# Use the official Python base image
FROM python:3.9.5

# Set the working directory in the container
WORKDIR /

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]