FROM drugilsberg/rdkit-ubuntu:latest
RUN apt-get update && apt-get install -y git
WORKDIR /toxsmi
# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# copy toxsmi
COPY . .
RUN pip3 install --no-deps .
CMD /bin/bash