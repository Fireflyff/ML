import fileinput

import tensorboard.compat.tensorflow_stub.tensor_shape
import torch
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
tf.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile

writer = SummaryWriter("../embedding")
embedded = torch.rand(30, 3)
meta = list(map(lambda x: x.strip(), fileinput.FileInput("../model_package/word_30.csv")))
writer.add_embedding(embedded, metadata=meta)
writer.close()
