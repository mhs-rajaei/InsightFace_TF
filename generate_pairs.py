#! encoding: utf-8

import os
import random

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.

    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """
    counter = 1

    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext


    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()


    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in os.listdir(self.data_dir):
            if name == ".DS_Store":
                continue

            a = []
            for file in os.listdir(self.data_dir + '\\' + name):
            # for file in os.listdir(self.data_dir):
                if file == ".DS_Store":
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    temp = random.choice(a).split("_")  # This line may vary depending on how your images are named.
                    # w = temp[0] + "_" + temp[1]
                    w = '_'.join(temp[:-1])
                    try:
                        l = random.choice(a).split("_")[-1].lstrip("0").rstrip(self.img_ext)
                        r = random.choice(a).split("_")[-1].lstrip("0").rstrip(self.img_ext)

                        print(w + " " + l + " " + r, 'counter: ', self.counter)

                        if 'William_Ford' in a:
                            print()


                        f.write(w + "\t" + l + "\t" + r + "\n")
                        self.counter += 1
                    except Exception as err:
                        print(err)
                        # print('Setting index to 1')
                        # l = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                        # l = random.choice(a).split.lstrip("0").rstrip(self.img_ext)
                        # r = random.choice(a).split("_")[1].lstrip("0").rstrip(self.img_ext)
                        # f.write(w + "\t" + l + "\t" + r + "\n")



    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store":
                continue

            remaining = os.listdir(self.data_dir)
            remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]
            # del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                for i in range(3):
                    # file1 = random.choice(os.listdir(self.data_dir + name))
                    file1 = random.choice(os.listdir(self.data_dir + '\\' + name))
                    # file2 = random.choice(os.listdir(self.data_dir + other_dir))
                    file2 = random.choice(os.listdir(self.data_dir + '\\' + other_dir))
                    if 'William_Ford' in file1 or 'William_Ford' in file2:
                        print()
                    if 'Yang_Hee_Kim' in file2:
                        print()
                    # f.write(name + "\t" + file1.split("_")[2].lstrip("0").rstrip(self.img_ext) + "\t")
                    f.write(name + "\t" + file1.split("_")[-1].lstrip("0").rstrip(self.img_ext) + "\t" + other_dir + "\t" + file2.split("_")[
                        -1].lstrip("0").rstrip(self.img_ext) + "\n")
                    print(name + " " + file1.split("_")[-1].lstrip("0").rstrip(self.img_ext) + ' ' + other_dir + ' ' +
                          file2.split("_")[-1].lstrip("0").rstrip(self.img_ext), 'counter:', self.counter)
                    self.counter += 1
        print()
                # f.write("\n")


if __name__ == '__main__':
    # data_dir = "my_own_datasets/"
    data_dir = r'F:\Documents\JetBrains\PyCharm\OFR\images\200END_lfw_160_test_Copy'
    pairs_filepath = r"F:\Documents\JetBrains\PyCharm\OFR\InsightFace_TF\data\pairs.txt"
    # img_ext = ".jpg"
    img_ext = ".png"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()

