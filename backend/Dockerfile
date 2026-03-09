#ligthweight python image
FROM python:3.11-slim

#set working directory
WORKDIR /app

#copies requirements -> container
COPY requirements.txt .

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#copy app folder -> container
COPY app ./app

#uses port 8080
EXPOSE 8080

#starts FASTAPI + make it accessible
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
