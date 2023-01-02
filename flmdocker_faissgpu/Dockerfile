FROM gcr.io/kaggle-gpu-images/python

RUN pip uninstall torch -y && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 && pip install pylcs cdifflib
RUN conda install faiss-gpu -y
