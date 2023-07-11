# Use an official Python runtime as a parent image
#FROM nvidia/cuda:11.7.0-base-ubuntu20.04
FROM pytorch/pytorch:latest

# Set the working directory
WORKDIR /app

# Install git, wget, build-essential
RUN apt-get update && apt-get install -y git wget build-essential

ENV TZ=Europe/Oslo \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install libportaudio2 libportaudiocpp0 portaudio19-dev libdb-dev

# Clone the repository
#RUN git clone https://github.com/vaaale/tortoise-tts-fast.git /app/tortoise-tts-fast
COPY . /app/tortoise-tts-fast/
RUN rm /app/tortoise-tts-fast/config.db

# Change the working directory to the tortoise-tts-fast directory
WORKDIR /app/tortoise-tts-fast

# Install the necessary packages
#RUN conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -c nvidia && \
#    conda install -c anaconda gdbm && \
#    pip install -e . && \
#    pip install git+https://github.com/152334H/BigVGAN.git && \
#    pip install streamlit

RUN python -m pip install berkeleydb
RUN python -m pip install -r requirements.txt
RUN python -m pip install .
# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV NAME tortoise-tts

# List the contents of the /app directory
RUN ls -al /app/
RUN ls -al /app/tortoise-tts-fast/

# Run the application
ENV PYTHONPATH=$PYTHONPATH:/app/tortoise-tts-fast/
CMD ["streamlit", "run", "scripts/app.py"]
