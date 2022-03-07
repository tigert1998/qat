from networks.image_classification_network import ImageClassificationNetwork


class CNNWrapper(ImageClassificationNetwork):
    def __init__(self, cnn, device):
        super().__init__(device)
        self.cnn = cnn
        self.cnn.to(self.device)

    def forward(self, x):
        return self.cnn.forward(x)
