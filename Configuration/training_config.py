"""configuration of training"""
import torch
from Configuration.environment_config import environment
from utils.train_utils import loss_param
from torch.utils.data import DataLoader

min_SNR = 5
max_SNR = 20

# learning_rate = 0.001  # Over fitting (maybe because of high learning rate)
learning_rate = 0.0001
batch_size = 500  # 5% of training set
weight_decay = 0

# train_data_path = f"G:/Shared drives/Track-Before-Detect/Track-Before-Detect/TBDViterbiNet/Data/training_data/{SNR}_SNR_train_data"
# valid_data_path = f"G:/Shared drives/Track-Before-Detect/Track-Before-Detect/TBDViterbiNet/Data/training_data/{SNR}_SNR_valid_data"
train_data_path = f"N:/6311/6311_Users/Noam/Multiple targets/Data/Training data/{min_SNR}-{max_SNR}_SNR_train_data"
valid_data_path = f"N:/6311/6311_Users/Noam/Multiple targets/Data/Training data/{min_SNR}-{max_SNR}_SNR_valid_data"
checkpoint_path = f"{min_SNR}-{max_SNR}_SNR_stats"

training_loss_param_list = [
    loss_param(
        epochs=50,
        environment=environment,
        ce_weight=0.5,
        frame_weight=0.0),
    loss_param(
     epochs=50,
     environment=environment,
     ce_weight=0.5,
     frame_weight=0.0
    ),
    loss_param(
     epochs=50,
     environment=environment,
     ce_weight=0.5,
     frame_weight=0.0
    )
]  # epochs are 1% of training set

valid_loss_param = loss_param(
    epochs=0,
    environment=environment,
    ce_weight=0.0,
    frame_weight=0.0
)

train_loader = DataLoader(
    torch.load(train_data_path),
    batch_size=batch_size,
    shuffle=True)

valid_loader = DataLoader(
    torch.load(valid_data_path),
    batch_size=batch_size,
    shuffle=True)

train_param = {
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "train_loader": train_loader,
    "valid_loader": valid_loader,
    "train_loss_param_list": training_loss_param_list,
    "valid_loss_param": valid_loss_param,
    "checkpoint_path": checkpoint_path,
    "min_SNR": min_SNR,
    "max_SNR": max_SNR
}
