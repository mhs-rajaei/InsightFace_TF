from sklearn import svm as sk_svm
from sklearn.neighbors import KNeighborsClassifier as sk_knn
from sklearn.externals import joblib
from sklearn import preprocessing

import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

class Classifier(object):

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def train(self):
        pass

    def check_accuracy(self, model, label_lookup_dict):
        """
        check_accuracy - function to check the accuracy of the test responses

        args    self.test_labels - original array of labels in int form
                model - trained model
                label_lookup_dict - dict for easy lookup of int to label
                test_response - array of predicted labels from the svm test

        returns accuracy - percentage accuracy
        """

        #reshape train data for compatibility with svm
        reshaped_test_data = self.data_reshape(self.test_data)
        predictions = model.predict_proba(reshaped_test_data)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        class_names = []
        prev_class = 1
        for cls in self.test_labels:
            if prev_class != cls:
                class_names.append(cls)
                prev_class = cls

        print("===================Prediction Table===================")
        print("%s  %s: %s \t %s" % ("test_num", "predicted face", "confidence", "expected face"))
        print("======================================================")
        for i in range(len(best_class_indices)):
            expected_face = label_lookup_dict[str(self.test_labels[i])]
            predicted_face = label_lookup_dict[str(class_names[best_class_indices[i]])]
            #ensure nice print formatting
            if len(predicted_face) > 13:
                predicted_face = predicted_face[:13]
            print('%4d  %s: %.3f \t expected: %s' % (i, predicted_face , best_class_probabilities[i], expected_face))

        accuracy = np.mean(np.equal(best_class_indices, self.test_labels))
        print("======================================================")
        print('Overall Accuracy: %.3f' % accuracy)

        return accuracy

    def data_reshape(self, data):
        """
        data_reshape - function to reshape the data to work with libsvm

        args    data - dataset to reshape

        return reshaped data - reshaped dataset
        """
        data_size = len(data)
        reshaped_data = data.reshape(data_size, -1)
        return reshaped_data

    def data_scale(self, data, skip_scale):
        """
        data_scale - normalise data points to a scale of 0-1

        arg     data - dataset to normalise

        return scaled_data - scaled version of dataset dataset
        """
        if skip_scale == False:
            scaled_data = preprocessing.normalize(data, norm='l1')
            return scaled_data
        elif skip_scale == True:
            return data

class SVM_Classifier(Classifier):

    def __init__(self, train_data, train_labels, test_data, test_labels):
        super().__init__(train_data, train_labels, test_data, test_labels)

    def train(self):
        """
        train - function to run the training on the svm

        args    self.train_data - data set for training (in numpy array form)
                self.train_labels - labels for the data set in int form

        returns svm - svm to test and run
        """
        #set up svm
        svm = sk_svm.SVC(kernel='linear', C=10, probability=True)

        #reshape train data for compatibility with svm
        reshaped_train_data = self.data_reshape(self.train_data)

        #scale train data for improved performance
        scaled_train_data = self.data_scale(reshaped_train_data, skip_scale=False)

        #train svm
        print("Starting to train svm...")
        svm.fit(scaled_train_data, self.train_labels)
        print("SVM training finished...")

        joblib.dump(svm, "latest_svm.pkl")

        return svm

class KNNClassifier(Classifier):
    """
    An implementation of K Nearest Neighbours classifier from scikit learn.
    """

    def __init__(self, train_data, train_labels, test_data, test_labels):
        super().__init__(train_data, train_labels, test_data, test_labels)

    def train(self):
        """
        train - function to run to train the model

        args    self.train_data - data set for training (in numpy array form)
                self.train_labels - labels for the data set in int form

        returns model - model to test and run
        """
        #set up svm
        model = sk_knn()

        #reshape train data for compatibility with svm
        reshaped_train_data = self.data_reshape(self.train_data)

        #train svm
        print("Starting to train KNN model...")
        model.fit(reshaped_train_data, self.train_labels)
        print("Training finished...")

        return model

class Neural_Classifier(Classifier):
    
    def __init__(self, train_data, train_labels, test_data, test_labels, num_classes):
        super().__init__(train_data, train_labels, test_data, test_labels)
        self.num_classes = num_classes

    def data_reshape(self, data):
        """
        data_reshape - function to reshape the data to work with keras

        args    data - dataset to reshape

        return reshaped data - reshaped dataset
        """
        data_size = len(data)
        reshaped_data = data.reshape(data_size, -1)
        print("DATA SHAPE")
        print(reshaped_data.shape)
        return reshaped_data


    def train(self):
         epochs = 25
         batch_size = 25
         self.train_data = self.data_reshape(self.train_data)
         input_shape = self.train_data.shape
         print(input_shape)

         model = Sequential()
         model.add(Dense(32, input_shape=(128,), activation='sigmoid'))
         model.add(Dense(self.num_classes, activation="softmax"))

         model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

         model.fit(self.train_data, self.train_labels, epochs=epochs, batch_size=batch_size)

         return model

    def check_accuracy(self, model):
        loss_and_metrics = model.evaluate(self.test_data, self.test_labels, batch_size = batch_size)
        return loss_and_metrics
