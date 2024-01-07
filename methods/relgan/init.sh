# relgan初始化

pip install --upgrade pip
pip install --upgrade pip

conda create -y -n relgan  python=3.5.2
conda activate relgan

pip install tensorflow==1.4
pip install numpy==1.14.1
pip install matplotlib==2.2.0
pip install scipy==1.0.0
pip install nltk==3.2.3
pip install tqdm==4.19.6
pip install requests


python -m nltk.downloader punkt


#####
# import nltk
# nltk.download('punkt')

# nohup python -u  keywords_relgan.py 0 0 > nohup.out 2>&1 &