# 📁 Dataset Directory Structure for Image Deraining

For training and testing, your dataset folder structure should look like this:

Datasets/
├── train/
│ └── Rain13K/
│ ├── input/ # Rainy images for training
│ └── target/ # Ground truth (clean) images for training
└── test/
├── Test100/
│ ├── input/ # Rainy images for testing
│ └── target/ # Ground truth (clean) images
├── Rain100H/
│ ├── input/ # Rainy images for testing
│ └── target/
├── Rain100L/
│ ├── input/ # Rainy images for testing
│ └── target/
├── Test1200/
│ ├── input/ # Rainy images for testing
│ └── target/
└── Test2800/
├── input/ # Rainy images for testing
└── target/