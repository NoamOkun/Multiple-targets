from Configuration.simulation_config import simulation_param
from utils.simulation_utils import Find_Accuracy
from Testing.test import test_model
from Graphs.plot_parts import plot_test_accuracy

track_loader = simulation_param["track_loader"]
tracker = simulation_param["tracker_model"]
name = simulation_param["tracker_name"]
ViterbiNet_acc = []

print(f'{name} test results:')
for (observation, label) in track_loader:
    observation = observation.squeeze(0)
    cheat_state = label[0]
    estimated_track = tracker(observation, cheat_state=cheat_state)
    acc = Find_Accuracy(estimated_track, label, name)
    # ViterbiNet_acc.append(acc[0])  # Real accuracy
    ViterbiNet_acc.append(acc[1])  # Soft accuracy

avg_ViterbiNet_acc = sum(ViterbiNet_acc)/len(ViterbiNet_acc)
print(f'ViterbiNet averaged accuracy: {avg_ViterbiNet_acc*100:.2f}%')

'''
    Testing the DNNTracker with the same test set for comparison with ViterbiNet
    to show that a kinematic model can improve accurate track recovery
'''
DNNTracker_acc, avg_DNNTracker_acc = test_model(track_loader, "DNNTracker")

filename = f'ViterbiNet {avg_ViterbiNet_acc*100:.2f}% vs DNNTracker {avg_DNNTracker_acc*100:.2f}% accuracy'

plot_test_accuracy(ViterbiNet_acc, DNNTracker_acc, filename)
