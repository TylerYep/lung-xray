# lung-xray
Stanford CS 271 Project

## TODO:
1. Change data to PyTorch Dataset
    - Fill in dataset.py MyDataset
2. Visualize/Explore data
    - viz.py
3. Add model
    - models/pnunet.py
4. Change loss/optimization
    - train.py


## How to Run
1. Start tensorboard using the `checkpoints/` folder with `tensorboard --logdir=checkpoints/`
2. Start and stop training using `python train.py --checkpoint=<checkpoint name>`. The code should automatically resume training at the previous epoch and continue logging to the previous tensorboard.
3. Run `python test.py --checkpoint=<checkpoint name>` to get final predictions.


# Directory Structure
- checkpoints/                  (Only created once you run train.py)
- data/
- metrics/
- models/
    - layers/
    - ...
- visualizers/
- args.py                       (Modify default hyperparameters here)
- dataset.py                    (Create Dataset here)
- metric_tracker.py
- models.py                     (You may opt to keep all your models in one place instead)
- preprocess.py                 (Do any preprocessing steps you want before loading the data)
- test.py
- train.py
- util.py
- viz.py                        (Create more visualizations if necessary)