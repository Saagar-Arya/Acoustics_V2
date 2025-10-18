# Acoustics V2
First draft of acoustics processing.
To be merged into 

## Running Files
### Neural Network
python src/nn_train.py --csv data/nn_sample_data.csv --epochs 100 --lr 1e-3

python src/nn_predict.py --ckpt artifacts/hydrophone_model.pt --csv data/nn_sample_data.csv --out probs.csv

### Read in Binary Files
#### Create executable (C++)
g++ -std=c++17 -O2 src/main.cpp -o readbin.exe
- TODO: Currently time values are slightly off from true time