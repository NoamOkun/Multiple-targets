import torch
from torch.utils.data import DataLoader
from Algorithms.ViterbiNet import ViterbiNet
from Configuration.viterbi_config import viterbinet_param


SNR = 20
Nt = 2  # Number of targets

# track_data_path = f"G:/Shared drives/Track-Before-Detect/Track-Before-Detect/TBDViterbiNet/Data/track_data/track_data_{SNR}_SNR"
track_data_path = f"N:/6311/6311_Users/Noam/Multiple targets/Data/Track data/track_data_{SNR}_SNR_{Nt}targets"
name = "ViterbiNet"


track_loader = DataLoader(torch.load(track_data_path))
viterbi_net = ViterbiNet(viterbinet_param=viterbinet_param)

simulation_param = {
    "track_loader": track_loader,
    "tracker_model": viterbi_net,
    "tracker_name": name
}

