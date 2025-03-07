import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseClassifier(nn.Module):
    """
    Base classifier with shared architecture for different branches.

    Args:
        input_dim (int): Input feature dimension.
        num_classes (int): Number of output classes.
    """
    def __init__(self, input_dim, num_classes):
        super(BaseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class SignalsClassifier(BaseClassifier):
    """
    Classifier for models using only the signal processing branch.

    Args:
        signals_dim (int): Dimension of signal features.
        num_classes (int): Number of output classes.
    """
    def __init__(self, signals_dim, num_classes):
        super(SignalsClassifier, self).__init__(signals_dim, num_classes)


class SequencesClassifier(BaseClassifier):
    """
    Classifier for models using only the sequence processing branch.

    Args:
        sequences_dim (int): Dimension of sequence features.
        num_classes (int): Number of output classes.
    """
    def __init__(self, sequences_dim, num_classes):
        super(SequencesClassifier, self).__init__(sequences_dim, num_classes)


class CombinedClassifier(nn.Module):
    """
    Classifier for models that use **both** the signal and sequence processing branches.

    Args:
        signals_dim (int): Dimension of signal features.
        sequences_dim (int): Dimension of sequence features.
        num_classes (int): Number of output classes.
    """
    def __init__(self, signals_dim, sequences_dim, num_classes):
        super(CombinedClassifier, self).__init__()
        input_dim = signals_dim + sequences_dim
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.bn(x)  # Batch normalization before passing through FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
