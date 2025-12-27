"""
Here we define the model that we want to use for the training 
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

class ECAPAEmotionClassifier(nn.Module):
    """
    A simple ECAPA + Dense head for utterance classification.
    """
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()

        #ECAPA backbone - takes input features (e.g., 80-dim mel bins), and produces a fixed length embedding
        self.encoder = ECAPA_TDNN(
            feat_dim, # no of mel channels/bins
            lin_neurons=256 # internal linear projection size
        )

        # Classification head: embedding → num_classes
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, features):
        """
        Docstring for forward

        :param features: tensor of shape (batch, time, feature_dim)
        """
        # produce an embedding of shape (batch, embedding_dim)
        embeddings = self.encoder(features)

        # classification logits
        logits = self.classifier(embeddings)

        return logits