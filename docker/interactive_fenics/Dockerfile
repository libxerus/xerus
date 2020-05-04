FROM firemarmot/xerus AS xerus

FROM quay.io/fenicsproject/stable:2019.1.0.r3
USER root
WORKDIR "/tmp/"

# Non-Python utilities and libraries
RUN apt-get -qq update && \
    apt-get -y install \
    libboost-filesystem-dev \
    libboost-system-dev \
    libboost-python-dev \
    liblapack* \
    libopenblas-dev \
    libiberty-dev \
    binutils-dev \
    zlib1g-dev

# Install Python2 based environment
RUN apt-get -y install \
    python-dev \
    python-numpy

# Install Python3 based environment
RUN apt-get -y install \
    python3-dev \
    python3-numpy

# Check Python versions
RUN python  -c'from __future__ import print_function; import sys; version = "{0}.{1}".format(*sys.version_info); print("python2-version:", version)'
RUN python3 -c'import sys; version = "{0}.{1}".format(*sys.version_info); print("python3-version:", version)'

# Clean apt
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create directories
RUN mkdir -p /usr/local/lib && \
	mkdir -p /usr/local/include && \
	mkdir -p /usr/local/lib/python2.7/site-packages && \
	mkdir -p /usr/local/lib/python3.6/site-packages

COPY --from=xerus /usr/local/include/suitesparse /usr/local/include/suitesparse
COPY --from=xerus /usr/local/lib/suitesparse /usr/local/lib/suitesparse
COPY --from=xerus /usr/local/include/xerus.h /usr/local/include
COPY --from=xerus /usr/local/include/xerus /usr/local/include
COPY --from=xerus /usr/local/lib/libxerus_misc.so /usr/local/lib
COPY --from=xerus /usr/local/lib/libxerus.so /usr/local/lib
COPY --from=xerus /usr/local/lib/python2.7/site-packages/xerus.so /usr/local/lib/python2.7/site-packages
COPY --from=xerus /usr/local/lib/python3.6/site-packages/xerus.so /usr/local/lib/python3.6/site-packages

# Install custom Python2 packages
RUN pip install cython && pip install ttpy matplotlib2tikz joblib

# Set environment variables
USER fenics
RUN mkdir -p $(python -m site --user-site) && \
	echo /usr/local/lib/python2.7/site-packages/ >$(python -m site --user-site)/xerus.pth && \
	mkdir -p $(python3 -m site --user-site) && \
	echo /usr/local/lib/python3.6/site-packages/ >$(python3 -m site --user-site)/xerus.pth
ENV LD_LIBRARY_PATH "/usr/local/lib/:/usr/local/lib/suitesparse:$LD_LIBRARY_PATH"

WORKDIR "/home/fenics/"
USER root
