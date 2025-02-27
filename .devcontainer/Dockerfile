FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Install software-properties-common to enable adding PPAs
RUN apt-get update && apt-get install -y software-properties-common

# Add the deadsnakes PPA for Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.11 and its development headers
RUN apt-get update && apt-get install -y python3.11 python3.11-dev

# Install pip for Python 3.11 using get-pip.py
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11