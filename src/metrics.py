import torch

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update_batch(self, predicted, labels):
        with torch.no_grad():
            self.correct += (predicted == labels).sum().item()
            self.total += labels.shape[0]

    def get_metrics_summary(self):
        accuracy = self.correct/self.total
        return accuracy