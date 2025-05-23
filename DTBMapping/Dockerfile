# Base image with TensorFlow and Python 3
FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

# 1. Update and Install System Dependencies
RUN apt-get update

# Install dependencies for opencv
RUN apt-get install -y ffmpeg libsm6 libxext6

# Install dependencies for GDAL (Geospatial Data Abstraction Library)
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL

# 2. Python Libraries for Geospatial and Image Processing
RUN pip install opencv-python
RUN pip install --upgrade geopandas
RUN pip install --upgrade rasterio
RUN pip install fiona
RUN pip install tifffile
RUN pip install whitebox

# Install splitraster without numpy version conflict
RUN pip install imageio==2.15.0
RUN pip install splitraster

# Install RVT for rendering visibility tool
RUN pip install rvt-py

# 3. Python Libraries for Machine Learning and Deep Learning
RUN pip install torch torchvision
RUN pip install --upgrade scikit-learn
RUN pip install --upgrade optuna
RUN pip install --upgrade xgboost
RUN pip install --upgrade quantile-forest
RUN pip install --upgrade numpy
RUN pip install --upgrade shap
RUN pip install --upgrade numba
RUN pip install pykrige

# 4. Python Libraries for Data Visualization
RUN pip install --upgrade matplotlib
RUN pip install plotly
RUN pip install --upgrade seaborn
RUN pip install altair

# 5. Jupyter and Interactive Widgets
RUN pip install jupyterlab
RUN pip install ipywidgets

# 6. Scientific Libraries
RUN pip install scipy

# 7. Create Mount Points for Data and Source Code
# RUN mkdir /workspace/data
# RUN mkdir /workspace/code
# RUN mkdir /workspace/temp
# RUN mkdir /workspace/temp_inference
# RUN mkdir /workspace/repo

# Copy code into container and set the working directory
# COPY . /workspace/repo/
# WORKDIR /workspace/code
