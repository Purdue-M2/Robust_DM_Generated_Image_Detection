import pandas as pd
import io
from PIL import Image
import numpy as np

def get_data(name):
    df = pd.read_parquet(name)
    img_id = df['id'].tolist()
    original_prompt = df['original_prompt'].tolist()
    positive_prompt = df['positive_prompt'].tolist()
    img_url = df['img_url'].tolist()
    # Assuming the images are stored as bytes in these columns
    image_gen0 = [row for row in df['image_gen0']]
    image_gen1 = [row for row in df['image_gen1']]
    image_gen2 = [row for row in df['image_gen2']]
    image_gen3 = [row for row in df['image_gen3']]
    return img_id, original_prompt, positive_prompt, img_url, image_gen0, image_gen1, image_gen2, image_gen3


