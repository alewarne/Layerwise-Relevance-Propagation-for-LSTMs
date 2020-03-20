import numpy as np
from lstm_network import LSTM_network

if __name__ == '__main__':
    n_hidden = 300
    n_embedding = 200
    n_classes = 2
    batch_size = 2
    net = LSTM_network(n_hidden, n_embedding, n_classes, batch_size)
    input = np.random.randn(batch_size,n_embedding)
    net.forward(input)
