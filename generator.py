import os
import yaml
from tqdm import tqdm

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_config():
    with open("generator.cfg", "r", encoding="utf-8") as fp:
        return yaml.load(fp, yaml.FullLoader)

def load_model():
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform

    return midas, transform

def generate_depth_info(model, transform, img_path, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()

def main():
    config = load_config()
    model, transform = load_model()

    source_path = config["source_path"]
    dest_path = os.path.join(config["dest_path"], "depth")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    for dataset_type in os.listdir(source_path):
        if not os.path.isdir(os.path.join(dest_path, dataset_type)):
            os.mkdir(os.path.join(dest_path, dataset_type))

        source_image_path = os.path.join(source_path, dataset_type, "images")
        dest_image_path = os.path.join(dest_path, dataset_type, "images")

        source_image_dir_list = os.listdir(source_image_path)

        if not os.path.isdir(dest_image_path):
            os.mkdir(dest_image_path)

        for image_dir in tqdm(source_image_dir_list):
            source_image_dir_path = os.path.join(source_image_path, image_dir)
            dest_image_dir_path = os.path.join(dest_image_path, image_dir)

            if not os.path.isdir(dest_image_dir_path):
                os.mkdir(dest_image_dir_path)

            for image_fn in os.listdir(source_image_dir_path):

                try:
                    depth_img = generate_depth_info(
                        model=model,
                        transform=transform,
                        img_path=os.path.join(source_image_dir_path, image_fn),
                        device=device)

                    np.save(os.path.join(dest_image_dir_path, image_fn[:-4]), depth_img)
                except Exception as e:
                    pass

if __name__ == "__main__":
    main()

