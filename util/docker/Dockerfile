# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Docker file for OpenTitan side channel analysis and fault injection setup.

FROM ubuntu:22.04

LABEL version="0.2"
LABEL description="OpenTitan SCA/FI image"
LABEL maintainer="nasahlpa@lowrisc.org"

ARG USER_NAME=ot
ARG MOUNT_DIR=/repo
ARG TIME_ZONE=America/New_York 
# This is convenient if we want to switch to different python version.
ARG PYTHON=python3.10
ARG VENV_PATH=/opt/venv
ARG GCC_VERSION=9

# Use bash as the default shell.
SHELL ["/bin/bash", "-c"]

# Imstall required packages.
# git-lfs: https://github.com/git-lfs/git-lfs/wiki/Installation#docker-recipes
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
        git \
        curl \
        ca-certificates \
        screen \
        locales \
        tzdata && \
    curl -fsSL https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | DEBIAN_FRONTEND="noninteractive" bash && \
    apt-get update && \
    apt-get install --no-install-recommends -y git-lfs

# Set locale and time zone.
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
RUN ln -fs /usr/share/zoneinfo/"${TIME_ZONE}" /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata

# Install the CI version of gcc and g++
RUN apt-get install --no-install-recommends -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 90 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} 90

# Chipwhisperer dependencies.
# https://chipwhisperer.readthedocs.io/en/latest/linux-install.html#required-packages
RUN DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
    libusb-dev \
    libusb-1.0-0-dev \
    make

# Python virtual environment and dependencies.
# Note: Permissions are relaxed so that the user created in the entrypoint can also use `pip`.
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    "${PYTHON}" \
    "${PYTHON}"-dev \
    "${PYTHON}"-distutils \
    "${PYTHON}"-venv
RUN "${PYTHON}" -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}"/bin:"${PATH}"
ENV VIRTUAL_ENV="${VENV_PATH}"
COPY python-requirements.txt /tmp/python-requirements.txt
COPY python-requirements-lint.txt /tmp/python-requirements-lint.txt
RUN pip install --upgrade pip "setuptools<66.0.0" && \
    pip install -r /tmp/python-requirements.txt && \
    chmod -R o=u "${VENV_PATH}";

# Cleanup
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/doc/*

# Entrypoint
COPY util/docker/docker_entrypoint.sh /docker_entrypoint.sh
RUN echo "exec /docker_entrypoint.sh" > /docker_entrypoint_wrapper.sh
RUN chmod +x /docker_entrypoint.sh /docker_entrypoint_wrapper.sh
ENTRYPOINT /docker_entrypoint_wrapper.sh
