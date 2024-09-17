from Configuration.viterbi_config import viterbinet_param
from Configuration.environment_config import environment
from utils.train_utils import get_one_hot, get_accuracy
import random


def test_model(track_loader, name):
    """
    Testing a DNN model
    Args:
        track_loader: A set of tracks for testing
        name: The name of the model we're testing
    Returns: A list of the accuracy for each track estimate
    """
    model = viterbinet_param["dnn_tracker"]
    bbox_param = environment.bbox_param
    bbox_acc = 0.0  # Bounding box accuracy
    test_acc_save = []
    print(f'\n{name} test results:\n')
    for (observation, label) in track_loader:
        observation = observation.squeeze(0)  # observation.shape() = [50, 1, 200, 64)
        num_frames = observation.shape[0]  # Length of the track
        for i in range(num_frames):
            z_k = observation[i]  # One frame from the track
            center = [label[i][0] + random.randint(-bbox_param[0] + 1, bbox_param[0]),
                      label[i][1] + random.randint(-bbox_param[1] + 1, bbox_param[1])]  # Center of search region TODO from train_utils.py and not ViterviNet
            bbox = environment.get_bbox(center)  # Set a bounding box
            one_hot, true_label = get_one_hot(environment, label[i], bbox)
            x_bbox = model(z_k, bbox, restore=False).view(-1)  # Estimate the state TODO restore?
            bbox_acc += get_accuracy(x_bbox, true_label)
        bbox_acc /= num_frames  # Track accuracy
        print(f'{name} accuracy =  {bbox_acc * 100:.2f}%')
        test_acc_save.append(bbox_acc)
    average = sum(test_acc_save) / len(test_acc_save)
    print(f'{name} average accuracy =  {average * 100:.2f}%')
    return test_acc_save, average
