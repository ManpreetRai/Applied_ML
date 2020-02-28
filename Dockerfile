FROM jupyter/pyspark-notebook

RUN python -m pip install --upgrade pip

RUN python -m pip install torch torchvision