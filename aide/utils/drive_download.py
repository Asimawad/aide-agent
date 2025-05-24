import gdown
# https://drive.google.com/file/d/1Jxbvpkor3e7t1T-VW6JyKTlN2SdL1ypS/view?usp=sharing
# Replace with your actual Google Drive file ID
file_id = '1Jxbvpkor3e7t1T-VW6JyKTlN2SdL1ypS'
output = 'data.zip'  # Desired local filename

url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, output, quiet=False)

# unzip the data.zip file
import zipfile
with zipfile.ZipFile('data.zip', 'r') as zip_ref:
    zip_ref.extractall('data')