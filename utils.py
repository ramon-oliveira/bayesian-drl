

def preprocessing(I):
    # copy from karpathy
    # I = I[35:195] # crop score bar
    I = I[::2, ::2, 0] # down sampling
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to
    return I
