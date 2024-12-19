import os
import shutil
import gdown

FILE_ID = "13pS_4Lk5v3Gq-0UtDE8nUgGl4KufADhx"

def download():
    gdown.download(id=FILE_ID, output="gan_best_model.pth")
    
    os.makedirs("src/model_weights", exist_ok=True)
    shutil.move("gan_best_model.pth", "src/model_weights/gan_best_model.pth")


if __name__ == "__main__":
    download()
