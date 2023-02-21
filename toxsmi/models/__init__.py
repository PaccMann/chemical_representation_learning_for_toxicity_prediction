from .cnn import CNN
from .rnn import RNN
from .dense import Dense
from .mca import MCAMultiTask

# More models could follow
MODEL_FACTORY = {'mca': MCAMultiTask, 'dense': Dense, 'cnn': CNN, 'rnn': RNN}
