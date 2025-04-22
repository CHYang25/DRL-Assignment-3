class Base_DQN_Agent(object):

    def __init__(self):
        super().__init__()

    def act(self, observation):
        raise NotImplementedError
    
    def train(self, batch):
        raise NotImplementedError
