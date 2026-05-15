import urllib.request
import zipfile
import os

url = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
zip_name = "att_faces.zip"
data_path = "face_dataset"

if not os.path.exists(zip_name):
    print("正在自动下载ORL人脸数据集...")
    urllib.request.urlretrieve(url, zip_name)
    print("下载完成")

if not os.path.exists(data_path):
    print("正在解压数据集...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall()
    os.rename("att_faces", data_path)
    print("解压并重命名完成")
else:
    print(f"数据集已存在于 {data_path}")