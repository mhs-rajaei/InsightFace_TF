#! encoding: utf-8

import os
import random


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.

    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """
    counter = 1

    def __init__(self, data_dir, pairs_filepath, img_ext, num_random_images_per_folder):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext
        self.num_random_images_per_folder = num_random_images_per_folder

        if os.name == 'nt':
            self.separator = "\\"
        else:
            self.separator = "/"

        self.remaining = []
        for name in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, name)):
                self.remaining.append(name)

    def update_remaining(self):
        self.remaining = []
        for name in os.listdir(self.data_dir):
            if os.path.isdir(os.path.join(self.data_dir, name)):
                self.remaining.append(name)

    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()

    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in self.remaining:
            a = []
            for file in os.listdir(os.path.join(self.data_dir, name)):
                if self.img_ext in file:
                    a.append(os.path.join(name, file))

            if a:
                with open(self.pairs_filepath, "a") as f:
                    for i in range(self.num_random_images_per_folder):
                        temp = random.choice(a).split(self.separator)  # This line may vary depending on how your images are named.
                        w = self.separator.join(temp[:-1])

                        l = random.choice(a).split(self.separator)[-1]
                        r = random.choice(a).split(self.separator)[-1]

                        print("For '" + os.path.join(self.data_dir, name) + "' and counter: ", self.counter, ', Match Pair:', w + " -> " + l
                              + ", " + r)

                        f.write(w + "\t" + l + "\t" + r + "\n")
                        self.counter += 1


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        tmptmp = ['00061', '10285', '00074', '10156', '10041', '20344', '10041', '20344', '10041', '20344', '10217', '20345', '20324', '20345', '20344',
                  '10268', '20345', '20481', '20394', '00074', '20412', '10014', '20436', '20412', '30604', '10218']
        for i, name in enumerate(self.remaining):
                self.update_remaining()
                del self.remaining[i]  # deletes the file from the list, so that it is not chosen again
                other_dir = random.choice(self.remaining)
                with open(self.pairs_filepath, "a") as f:
                    for _ in range(self.num_random_images_per_folder):

                        if name in tmptmp:
                            print()
                        if other_dir in tmptmp:
                            print()

                        temps_file_1 = os.listdir(os.path.join(self.data_dir, name))
                        if temps_file_1:
                            file1 = random.choice(temps_file_1)

                        temps_file_2 = os.listdir(os.path.join(self.data_dir, other_dir))
                        if temps_file_2:
                            file2 = random.choice(temps_file_2)

                        if temps_file_1 and temps_file_2:
                            if self.img_ext in file1 and self.img_ext in file2:
                                print("For '" + self.data_dir + "' and counter: ", self.counter, ', MisMatch Pair:',
                                      name + " " + file1.split(self.separator)[-1] + ' ' +
                                      other_dir + ' ' + file2.split(self.separator)[-1])

                                f.write(name + "\t" + file1.split(self.separator)[-1] + "\t" + other_dir + "\t" +
                                        file2.split(self.separator)[-1] + "\n")

                                self.counter += 1


if __name__ == '__main__':
    data_dir = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\All VIS+NIR_112"
    eval_pairs_path = os.path.join(PROJECT_PATH, 'data/All VIS+NIR_112_pairs_6.txt')
    pairs_filepath = eval_pairs_path
    # img_ext = ".jpg"
    img_ext = ".png"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext, num_random_images_per_folder=6)
    generatePairs.generate()

