FROM ubuntu:20.04

RUN  apt-get update
RUN apt-get install -y wget unzip
RUN rm -rf /var/lib/apt/lists/*
 

FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

# Install requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the contents of repository
COPY . .

# Expose port
EXPOSE 3000