For training and testing, your directory structure should look like this

Datasets/
├── train/
│   └── Rain13K/
│       ├── input/     # Rainy images for training
│       └── target/    # Ground truth (clean) images for training
└── test/
    ├── Test100/
    │   ├── input/     # Rainy images for testing
    │   └── target/    # Ground truth (clean) images
    ├── Rain100H/
    │   ├── input/
    │   └── target/
    ├── Rain100L/
    │   ├── input/
    │   └── target/
    ├── Test1200/
    │   ├── input/
    │   └── target/
    └── Test2800/
        ├── input/
        └── target/
