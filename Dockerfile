FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime as pytorch

ENV PROJECT_PATH /workspace/any-quantile

RUN date
RUN apt-get update && apt-get install -y locales && locale-gen en_US.UTF-8 && apt-get -y install git g++ zip unzip gnupg
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

RUN python -m pip install pip -U

# Install tini, which will keep the container up as a PID 1
RUN apt-get update && apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=0.19.0 && \
    curl -L "https://github.com/krallin/tini/releases/download/v0.19.0/tini_0.19.0.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Install AWS
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm awscliv2.zip && rm -rf aws
    
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y
    
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt -f https://download.pytorch.org/whl/torch_stable.html --ignore-installed PyYAML
RUN pip install psutil --upgrade 

# Install AWS
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install --update && \
    rm awscliv2.zip && rm -rf aws
    
RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py 

ENTRYPOINT [ "/usr/bin/tini", "--" ]

CMD ["jupyter", "notebook", "--allow-root"]

WORKDIR ${PROJECT_PATH}
