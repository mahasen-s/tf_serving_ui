FROM python:3.6

WORKDIR /usr/src/app

RUN apt update && apt install -y vim
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod 755 app.py
CMD ["python", "app.py"]
