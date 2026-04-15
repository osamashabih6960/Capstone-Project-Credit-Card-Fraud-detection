FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/power_transformer.pkl /app/models/power_transformer.pkl

RUN pip install -r requirements.txt

EXPOSE 5000

#local
#CMD ["python", "app.py"]

#Prod
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]