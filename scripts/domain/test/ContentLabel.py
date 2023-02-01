


class ContentLabel:
    def __init__(self, label):
        self.label = label


    def __eq__(self, other):
        assert isinstance(other, ContentLabel)
        return all(self.label == other.label)
