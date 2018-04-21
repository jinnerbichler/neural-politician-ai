FROM python:3.6-onbuild
MAINTAINER Johannes Innerbichler <j.innerbichler@gmail.com>

# setup proper configuration
ENV PYTHONPATH .

ENTRYPOINT ["python", "manage.py"]