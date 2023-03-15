# Set base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone code from remote repo
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Set port which container listens to at runtime
EXPOSE 8501

# Tell docker to check container is still working
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set entrypoints
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]