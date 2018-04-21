FROM python:3.6-jessie
MAINTAINER Johannes Innerbichler <j.innerbichler@gmail.com>

COPY frontend /frontend/

# setup proper configuration
ENV PYTHONPATH .

ENTRYPOINT ["python", "manage.py"]