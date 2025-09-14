import zipfile
with zipfile.ZipFile('data/raw/playground-series-s5e9.zip', 'r') as zip_ref:
    zip_ref.extractall(path='data/raw/')