conda install -y -c conda-forge notebook pandas matplotlib icecream pytorch-lightning youtokentome datasets transformers tokenizers
conda install -y -c pytorch torchaudio torchtext
jupyter notebook --no-browser --ip=0.0.0.0 --port=8080 --allow-root


# to download things from gdrive
# conda install -y -c conda-forge gdown
# gdown https://drive.google.com/uc?id=1d3Zb9heon2FFv3rZEW2YaI8F46aSkXba
# tar -xf cv-corpus-7.0-2021-07-21-cs.tar.gz