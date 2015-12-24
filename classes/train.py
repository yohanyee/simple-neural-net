from pipeline import *
from data import *
import sys

class Trainer(object):
    def __init__(self, DataObject, NetworkObject):
        self.Network = NetworkObject
        self.Data = DataObject
        self.iterations_trained = 0
        self.epochs_trained = 0
        self.validation_classification_rate = 0.
        self.learning_rate = 1.
        self.variable_learning = True
        self.acc_train = []
        self.acc_valid = []
        self.acc_test = []
        
    def get_train_classification_rate(self, print_to_screen=False, batch_size=100):
        correct = 0
        wrong = 0
        self.misclassified_train = []
        num_sets = min(batch_size, self.Data.train_num)
        elist = np.random.choice(range(self.Data.train_num), num_sets, replace=False)
        for e in elist:
            X = self.Data.X_train[e]
            t = self.Data.y_train[e]
            self.Network.infer(X)
            y = self.Network.OutputLayer.get_activations()[0]
            if y < 0.5:
                prediction = 0
            else:
                prediction = 1
            if t==prediction:
                correct += 1
            else:
                wrong += 1
                self.misclassified_train.append((X,t))
        train_classification_rate = float(correct)/num_sets
        if print_to_screen:
            logstr = str(correct) + " of " + str(num_sets) + " classified correctly."
            print(logstr)
            logstr = str(100*train_classification_rate) + "% classification rate on train dataset."
            print(logstr)
        else:
            return(train_classification_rate)
        
    def get_validation_classification_rate(self, print_to_screen=False, batch_size=100):
        correct = 0
        wrong = 0
        self.misclassified_validation = []
        num_sets = min(batch_size, self.Data.valid_num)
        elist = np.random.choice(range(self.Data.valid_num), num_sets, replace=False)
        for e in elist:
            X = self.Data.X_valid[e]
            t = self.Data.y_valid[e]
            self.Network.infer(X)
            y = self.Network.OutputLayer.get_activations()[0]
            if y < 0.5:
                prediction = 0
            else:
                prediction = 1
            if t==prediction:
                correct += 1
            else:
                wrong += 1
                self.misclassified_validation.append((X,t))
        validation_classification_rate = float(correct)/num_sets
        if print_to_screen:
            logstr = str(correct) + " of " + str(num_sets) + " classified correctly."
            print(logstr)
            logstr = str(100*validation_classification_rate) + "% classification rate on validation dataset."
            print(logstr)
        else:
            return(validation_classification_rate)
            
    def get_test_classification_rate(self, print_to_screen=False, batch_size=100):
        correct = 0
        wrong = 0
        self.misclassified_test = []
        num_sets = min(batch_size, self.Data.test_num)
        elist = np.random.choice(range(self.Data.test_num), num_sets, replace=False)
        for e in elist:
            X = self.Data.X_test[e]
            t = self.Data.y_test[e]
            self.Network.infer(X)
            y = self.Network.OutputLayer.get_activations()[0]
            if y < 0.5:
                prediction = 0
            else:
                prediction = 1
            if t==prediction:
                correct += 1
            else:
                wrong += 1
                self.misclassified_test.append((X,t))
        test_classification_rate = float(correct)/num_sets
        if print_to_screen:
            logstr = str(correct) + " of " + str(num_sets) + " classified correctly."
            print(logstr)
            logstr = str(100*test_classification_rate) + "% classification rate on test dataset."
            print(logstr)
        else:
            return(test_classification_rate)

class BackpropagationTrainer(Trainer):
    def __init__(self, DataObject, NetworkObject):
        super(BackpropagationTrainer, self).__init__(DataObject, NetworkObject)

    def study_example(self, input_train, target_train):
        self.Network.study(input_train, target_train)
        self.iterations_trained += 1

    def train_example(self, input_train, target_train):
        self.Network.learn(input_train, target_train, self.learning_rate)
        self.iterations_trained += 1
    
    def train_dataset(self, batch_update=True, batch_size=1000, train_batch_size=100, validation_batch_size=100, test_batch_size=100):
        training_inputs = self.Data.X_train
        training_targets = self.Data.y_train 
        num_sets = min(batch_size, self.Data.train_num)
        elist = np.random.choice(range(self.Data.train_num), num_sets, replace=False)
        logstr = "Training epoch: " + str(self.epochs_trained + 1) + ". "
        sys.stdout.write(logstr)
        sys.stdout.flush()
        pc_str_len = 0
        e = 1
        for example in elist:
            if batch_update:
                self.study_example(training_inputs[example], training_targets[example])
            else:
                self.train_example(training_inputs[example], training_targets[example])
            percent_complete = 100.*float(e)/num_sets
            percent_complete_string = "{:3.2f}".format(percent_complete) + "% complete."
            sys.stdout.write(pc_str_len*'\b')
            sys.stdout.write(pc_str_len*' ')
            sys.stdout.write(pc_str_len*'\b')
            sys.stdout.write(percent_complete_string)
            sys.stdout.flush()
            pc_str_len = len(percent_complete_string)
            e += 1
        if batch_update:
            self.Network.update(self.learning_rate)
        self.epochs_trained += 1
        validation_classification_rate = self.get_validation_classification_rate(batch_size=validation_batch_size)
        if (validation_classification_rate <= self.validation_classification_rate) and \
            ((self.validation_classification_rate - validation_classification_rate) < 0.011):
            self.Network.trained = True
        self.validation_classification_rate = validation_classification_rate
        if self.variable_learning:
            self.learning_rate = 2.01-2.*validation_classification_rate
            
        dcr = self.get_train_classification_rate(batch_size=train_batch_size)
        vcr = validation_classification_rate
        tcr = self.get_test_classification_rate(batch_size=test_batch_size)
        self.acc_train += [dcr]
        self.acc_valid += [vcr]
        self.acc_test += [tcr]
        logstr = "({:3.2f},{:3.2f},{:3.2f})% correct on (train,valid,test) patches.".format(100.*dcr, 100.*vcr, 100.*tcr)
        sys.stdout.write(pc_str_len*'\b')
        sys.stdout.write(pc_str_len*' ')
        sys.stdout.write(pc_str_len*'\b')
        sys.stdout.write(logstr + "\n")
        sys.stdout.flush()
    
    def train(self, epochs=None, batch_update=True, batch_size=1000, train_batch_size=100, validation_batch_size=100, test_batch_size=100, ignore_trained=True):
        self.Network.trained = False
        dcr = self.get_train_classification_rate(batch_size=train_batch_size)
        vcr = self.get_validation_classification_rate(batch_size=validation_batch_size)
        tcr = self.get_test_classification_rate(batch_size=test_batch_size)
        logstr = "Current classification rate: ({:3.2f},{:3.2f},{:3.2f})% correct on (train,valid,test) patches.".format(100.*dcr, 100.*vcr, 100.*tcr)
        print(logstr)
        if epochs is None:
            epochs = 1000
        for epoch in range(epochs):
            self.train_dataset(batch_update, batch_size, train_batch_size, validation_batch_size, test_batch_size)
            if (self.Network.trained and not ignore_trained) == True:
                break

class GeneticAlgorithm(Trainer):
    def __init__(self, DataObject, NetworkObject):
        super(BackpropagationTrainer, self).__init__(DataObject, NetworkObject)
        
        


