import copy
import numpy as np
from .ContentLabel import ContentLabel


class UniqueLabelAssigner:
    def __init__(self):
        self.unique_content_dict = {}


    def assgin(self, content_label: ContentLabel):
        assert isinstance(content_label, ContentLabel)

        if self.unique_content_dict != {}:
            for key, val in self.unique_content_dict.items():
                if content_label == val: return int(key)

        new_content_num = self.get_max_content_num() + 1
        self.unique_content_dict[str(new_content_num)] = content_label
        return int(new_content_num)


    def get_max_content_num(self):
        max = -1
        for key in self.unique_content_dict.keys():
            if int(key) > max: max = int(key)
        return max
