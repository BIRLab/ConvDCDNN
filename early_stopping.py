class EarlyStopping:
    def __init__(self, tolerance=5):
        self.tolerance = tolerance
        self.counter = 0
        self.min_loss = float('inf')
        self.early_stop = False

    def __call__(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.min_loss = float('inf')
        self.early_stop = False


__all__ = ['EarlyStopping']
