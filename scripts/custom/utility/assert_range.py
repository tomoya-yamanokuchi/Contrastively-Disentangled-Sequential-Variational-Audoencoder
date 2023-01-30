

def assert_range(x, min, max):
    assert x.min() >= min, print("[x.min(), min] = [{}, {}]".format(x.min(), min))
    assert x.max() <= max, print("[x.max(), max] = [{}, {}]".format(x.max(), max))