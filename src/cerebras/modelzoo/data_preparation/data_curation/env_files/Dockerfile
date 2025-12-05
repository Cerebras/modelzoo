# Use a slim image based on Debian Bookworm (glibc 2.36, libstdc++ from GCC 12)
FROM python:3.11-slim

# ensure UTF‑8 locale (optional but recommended)
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

# system‑level deps for building C/C++ extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \         
      cmake \  
      libmagic1 \                 
      wget \                    
      libstdc++6 \   
      slurm-wlm \               
      munge \                   
      libmunge2 \                        
    && rm -rf /var/lib/apt/lists/*

# copy in and install Python deps
COPY ./requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# if you use NLTK data, you can pre‑download here
# RUN python -m nltk.downloader punkt wordnet

# default to an interactive shell
CMD ["bash"]