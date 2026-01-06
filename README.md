# 🌅 Scenic Viber
A high-performance image segmentation tool using **Mask2Former** and **Swin-Large**.

## ✨ Features
* **State-of-the-Art:** Powered by Swin-Large Transformers.
* **Versatile:** Supports Semantic, Instance, and Panoptic segmentation.
* **Easy-to-use:** Simple CLI for processing folders of images.

## 🚀 Getting Started
### Installation
1. Clone the repo: `git clone https://github.com/user/scenic-viber`
2. Install dependencies: `pip install -r requirements.txt`

### Usage
```python
from scenic_viber import ViberModel

model = ViberModel(backbone="swin-large")
model.predict("landscape.jpg")
