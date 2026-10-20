FROM registry.access.redhat.com/ubi9/python-312:latest@sha256:f43a57e9a6a2243e333008219002b9f2b1661868a94199f5dec4d602e1501482
WORKDIR /opt/app-root

USER root

COPY . .

# Build argument to specify CPU or GPU build
ARG BUILD_TYPE=gpu

# Conditionally install CPU-specific torch
RUN if [ "$BUILD_TYPE" = "cpu" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch>=2.6.0; \
    fi
RUN ls
RUN pip install --no-cache-dir -e ".[remote]"

# Set XDG environment variables to use /tmp (always writable) for garak to write to
ENV XDG_CACHE_HOME=/tmp/.cache
ENV XDG_DATA_HOME=/tmp/.local/share
ENV XDG_CONFIG_HOME=/tmp/.config

# USER 1001

