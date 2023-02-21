from .cnn import CNN
from .dense import Dense
from .mca import MCAMultiTask
from .rnn import RNN

# More models could follow
MODEL_FACTORY = {"mca": MCAMultiTask, "dense": Dense, "cnn": CNN, "rnn": RNN}
