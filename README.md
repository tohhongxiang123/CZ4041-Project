# PetFinder.my - Pawpularity Contest

Predict the popularity of shelter pet photos

- https://www.kaggle.com/competitions/petfinder-pawpularity-score

# Setup

```
python -m venv venv
./venv/Scripts/activate

pip install matplotlib numpy pandas opencv-python tqdm timm pytorch-lightning==1.4.0
```

Then we will need to install pytorch

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Then, download the data, and place it within `/data`. You can download the data [here](https://www.kaggle.com/competitions/petfinder-pawpularity-score/data). Within the data directory, it should look something like this:

```
data
├── test
├── train
├── sample_submission.csv
├── test.csv
└── train.csv
```

# Resources

- https://towardsdatascience.com/custom-dataset-in-pytorch-part-1-images-2df3152895