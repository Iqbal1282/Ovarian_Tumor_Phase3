FROM radiomics/pyradiomics:CLI

# Install system dependencies needed for visualization and Jupyter
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    jupyterlab \
    pandas \
    opencv-python \
    matplotlib \
    seaborn \
    scikit-learn

# Set working directory
WORKDIR /workspace

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
