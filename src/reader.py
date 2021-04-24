"""Dataset reader and process"""

import os
import html
import string
import cv2 as cv


class Dataset:
    """Dataset class to read images and sentences from base (raw files)"""

    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.dataset = None
        self.partitions = ['train', 'valid', 'test']

    def read_partitions(self):
        """Read images and sentences from dataset"""

        dataset = getattr(self, f"_{self.name}")()

        if not self.dataset:
            self.dataset = dict()

            for y in self.partitions:
                self.dataset[y] = {'dt': [], 'gt': []}

        for y in self.partitions:
            self.dataset[y]['dt'] += dataset[y]['dt']
            self.dataset[y]['gt'] += dataset[y]['gt']


    def _persefone(self):

        source = os.path.join(self.source)
        img_path = os.path.join(source, 'lines/')
        pt_path = os.path.join(source, "partitions/")
        transcriptions = os.path.join(source, "transcriptions/")

        paths = {"train": open(os.path.join(pt_path, "training.lst")).read().splitlines(),
                 "valid": open(os.path.join(pt_path, "validation.lst")).read().splitlines(),
                 "test": open(os.path.join(pt_path, "test.lst")).read().splitlines()}

        gt = os.listdir(transcriptions)
        gt_dict = dict()
        dataset = dict()

        for index, x in enumerate(gt):
            text = " ".join(open(os.path.join(transcriptions, x)).read().splitlines())
            text = html.unescape(text).replace("<gap/>", "")
            gt_dict[os.path.splitext(x)[0]] = " ".join(text.split())

        for i in self.partitions:
            dataset[i] = {"dt": [], "gt": []}

            for line in paths[i]:
                path_current_img = os.path.join(img_path, f"{line}.png")
                dataset[i]['dt'].append(path_current_img)
                dataset[i]['gt'].append(gt_dict[line])

        return dataset

    @staticmethod
    def check_text(text):
        """Make sure text has more characters instead of punctuation marks"""

        strip_punc = text.strip(string.punctuation).strip()
        no_punc = text.translate(str.maketrans("", "", string.punctuation)).strip()

        if len(text) == 0 or len(strip_punc) == 0 or len(no_punc) == 0:
            return False

        punc_percent = (len(strip_punc) - len(no_punc)) / len(strip_punc)

        return len(no_punc) > 2 and punc_percent <= 0.1
