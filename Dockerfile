# Use official Python image
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files
COPY requirements.txt .
COPY . .

RUN pip install setuptools --upgrade
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port (default: 8501)
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "./streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
