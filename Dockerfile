FROM ubuntu:22.04

############################ ns3 part

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"

RUN apt-get update && \
    apt-get install -y \
        git \
        mercurial \
        gcc \
        g++ \
        vim \
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        qtbase5-dev \
        qtchooser \
        qt5-qmake \
        qtbase5-dev-tools \
        gir1.2-goocanvas-2.0 \
        python3-gi \
        python3-gi-cairo \
        python3-pygraphviz \
        gir1.2-gtk-3.0 \
        ipython3 \
        autoconf \
        cvs \
        bzr \
        unrar \
        gdb \
        valgrind \
        uncrustify \
        flex \
        bison \
        libfl-dev \
        tcpdump \
        gsl-bin \
        libgsl-dev \
        sqlite \
        sqlite3 \
        libsqlite3-dev \
        libxml2 \
        libxml2-dev \
        cmake \
        libc6-dev \
        libc6-dev-i386 \
        libclang-dev \
        llvm-dev \
        automake \
        libgtk2.0-0 \
        libgtk2.0-dev \
        vtun \
        lxc \
        libboost-all-dev \
    	wget \
    	libgtk-3-dev \
    	ccache \
        curl
    	
RUN pip install --user cxxfilt cppyy

RUN mkdir -p /usr/ns3
WORKDIR /usr 

RUN wget https://www.nsnam.org/release/ns-allinone-3.36.1.tar.bz2  && \
    tar -jxvf ns-allinone-3.36.1.tar.bz2
    
RUN cd ns-allinone-3.36.1/ns-3.36.1/src && wget https://gitlab.com/cttc-lena/nr/-/archive/v2.2/nr-v2.2.tar.bz2 && \
    tar -jxvf nr-v2.2.tar.bz2 && mv nr-v2.2 nr && ls

RUN cd ns-allinone-3.36.1 && ls && ./build.py --enable-examples

RUN ln -s /usr/ns-allinone-3.36.1/ns-3.36.1/ /usr/ns3/

RUN apt-get clean && \
    rm -rf /var/lib/apt && \
    rm ns-allinone-3.36.1.tar.bz2 && \
    rm ns-allinone-3.36.1/ns-3.36.1/src/nr-v2.2.tar.bz2

ENV PATH="${PATH}:$/usr/ns3/ns-3.36.1"


################# neural network part


WORKDIR /app

ENV POETRY_HOME=/opt/poetry
ENV PATH="${PATH}:$POETRY_HOME/bin"
RUN curl -sSL https://install.python-poetry.org | python3 -

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

RUN poetry run poe addpytorch

ENV PATH="${PATH}:/usr/ns3/ns-3.36.1"

COPY . .

CMD poetry run poe webserver
