FROM python:3.8-slim-buster

RUN pip3 install --upgrade pip
RUN pip3 install flask
RUN pip3 install joblib
RUN pip3 install pandas
RUN pip3 install scikit-learn==0.22.1
RUN pip3 install gunicorn
RUN pip3 install requests
RUN pip3 install uuid
RUN pip3 install google-cloud-pubsub
RUN pip3 install google-cloud-storage
RUN pip3 install toto-pubsub-nicolasances
RUN pip3 install toto-logger-nicolasances
RUN pip3 install totoml==2.2.2

COPY . /app/

WORKDIR /app/

ENV TOTO_TMP_FOLDER=/modeltmp
ENV PYTHONUNBUFFERED=TRUE

CMD gunicorn --bind 0.0.0.0:8080 wsgi:app --enable-stdio-inheritance --timeout 3600 --workers=2