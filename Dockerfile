FROM mcr.microsoft.com/playwright/python:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Crawl4AI via pip and run its setup command
RUN pip install -U crawl4ai
RUN crawl4ai-setup
RUN playwright install

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8081"]
