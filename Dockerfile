FROM python:3.10.10

# Run updates and install ffmpeg
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install the requirements
COPY ./requirements.txt /requirements.txt

# Pip install the dependencies
RUN pip install --upgrade pip 
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the current directory contents into the container at /app
COPY main.py /app/main.py

# Set the working directory to /app
WORKDIR /app

# Expose port 8000
EXPOSE 8000

# Run the app
CMD uvicorn main:app --host 0.0.0.0
