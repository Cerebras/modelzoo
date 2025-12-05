FROM nvidia/cuda:12.5.1-devel-ubuntu24.04
RUN apt-get update && apt-get install python3-pip -y
RUN pip install --break-system-packages --extra-index-url https://pypi.nvidia.com "nemo-curator[cuda12x]==0.8.0rc3.dev0"
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /root/.cache/pip
