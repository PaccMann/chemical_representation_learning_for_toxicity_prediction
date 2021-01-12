from .mca import MCAMultiTask
from .dense import Dense
from .cnn import CNN
from .rnn import RNN

# More models could follow
MODEL_FACTORY = {'mca': MCAMultiTask, 'dense': Dense, 'cnn': CNN, 'rnn': RNN}
