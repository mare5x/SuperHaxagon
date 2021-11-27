from model import *

def in_range(lo, hi, x):
    return x >= lo and x < hi 

def test_pick_action():
    model = SupaDQN()
    model.set_is_learning(False)
    s = list(torch.rand(INPUT_SIZE))
    assert in_range(0, len(model.actions_tr), model.pick_action(s))
    model.set_is_learning(True) 
    assert in_range(0, len(model.actions_tr), model.pick_action(s))

def test_step():
    model = SupaDQN()

    model.set_is_learning(False)
    for _ in range(4 * model.batch_size):
        s = list(torch.rand(INPUT_SIZE))
        assert in_range(0, len(model.actions_tr), model.step(s, reward=random.random() - 0.5, done=random.random() < 0.5))
        
    model.set_is_learning(True)
    for _ in range(4 * model.batch_size):
        s = list(torch.rand(INPUT_SIZE))
        assert in_range(0, len(model.actions_tr), model.step(s, reward=random.random() - 0.5, done=random.random() < 0.5))

