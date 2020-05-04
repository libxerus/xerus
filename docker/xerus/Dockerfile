FROM firemarmot/suitesparse
USER root
WORKDIR "/tmp/"

# Check Python versions
RUN python  -c'from __future__ import print_function; import sys; version = "{0}.{1}".format(*sys.version_info); print("python2-version:", version); raise SystemExit(version != "2.7")'
RUN python3 -c'import sys; version = "{0}.{1}".format(*sys.version_info); print("python3-version:", version); raise SystemExit(version != "3.6")'

# Install Xerus
RUN git clone https://git.hemio.de/xerus/xerus.git
WORKDIR "/tmp/xerus/"
RUN git checkout vmc && \
	git pull && \
	git submodule update --init --recursive
RUN cp ./docker/xerus/config.mk . && \
	mkdir -p /usr/local/lib && \
	mkdir -p /usr/local/include && \
	mkdir -p /usr/local/lib/python2.7/site-packages && \
	mkdir -p /usr/local/lib/python3.6/site-packages && \
	make python2 python3 -j $(cat /tmp/NCORES_1)
RUN ls /usr/local/lib/python3.6/ && \
	make install -j $(cat /tmp/NCORES_1)

RUN mkdir -p $(python -m site --user-site) && \
	echo /usr/local/lib/python2.7/site-packages/ >$(python -m site --user-site)/xerus.pth && \
	mkdir -p $(python3 -m site --user-site) && \
	echo /usr/local/lib/python3.6/site-packages/ >$(python3 -m site --user-site)/xerus.pth
ENV LD_LIBRARY_PATH "/usr/local/lib/:$LD_LIBRARY_PATH"
