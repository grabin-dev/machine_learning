import numpy as np
np.random.seed(42)

####################################################################################################
#                                            Part A
####################################################################################################

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_instances = self.dataset[self.dataset.iloc[:,-1]==self.class_value]

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.class_instances.shape[0] / self.dataset.shape[0]
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        mean = self.class_instances.mean()
        std = self.class_instances.std()
        return normal_pdf(x, mean, std).prod()
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)
    
class MultiNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset from which to compute the mean and mu (Numpy Array).
        - class_value : The class to calculate the mean and mu for.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_instances = self.dataset[self.dataset.iloc[:,-1]==self.class_value]

    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.class_instances.shape[0] / self.dataset.shape[0]

    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        cov = self.class_instances.iloc[:,:-1].cov()
        mean = self.class_instances.iloc[:,:-1].mean(axis=0).values
        p = multi_normal_pdf(x.values,mean,cov)
        return p
        
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_prior() * self.get_instance_likelihood(x)
    
    

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    exponent = np.exp(-0.5 * (((x - mean)/ std) ** 2))
    fraction = 1 / np.sqrt(2 * np.pi * (std ** 2))
    return  fraction * exponent
    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variante normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    from math import pi
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    d = cov.shape[0]
    num1 = ((2 * pi) ** (-d / 2))* (det ** -0.5)
    exponent = np.exp(-0.5 * (x - mean).T @ cov_inv @ (x - mean))
    return exponent * num1
    

####################################################################################################
#                                            Part B
####################################################################################################
EPSILLON = 1e-6 # == 0.000001 It could happen that a certain value will only occur in the test set.
                # In case such a thing occur the probability for that value will EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with la place smoothing.
        
        Input
        - dataset: The dataset from which to compute the probabilites (Numpy Array).
        - class_value : Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        self.class_instances = dataset[dataset.iloc[:,-1] == class_value].iloc[:,:-1]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        return self.class_instances.shape[0] / self.dataset.shape[0]
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        vj = {
          'age' : 9,
          'menopause' : 3,
          'tumor-size': 12,
          'inv-nodes' : 13,
          'node-caps' : 2,
          'deg-malig' : 3,
          'breast' : 2,
          'breast-quad' : 5,
          'irradiat' : 2
        }

        prob = 1.0
        ni = self.class_instances.shape[0]
        for att in x.index:
            attribute_value = x[att]
            nij = self.class_instances[self.class_instances[att] == attribute_value]
            if nij.empty:
               nij = EPSILLON
            else:
                nij = nij.shape[0]
            prob *= (nij + 1) / (ni + vj[att])
        return prob
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.get_instance_likelihood(x) * self.get_prior()

    
####################################################################################################
#                                            General
####################################################################################################            
class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a postreiori classifier. 
        This class will hold 2 class distribution, one for class 0 and one for class 1, and will predicit and instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
        
        Input
            - An instance to predict.
            
        Output
            - 0 if the posterior probability of class 0 is higher 1 otherwise.
        """
        p0 = self.ccd0.get_instance_posterior(x)
        p1 = self.ccd1.get_instance_posterior(x)
        if p1 < p0:
            return 0
        return 1
    
def compute_accuracy(testset, map_classifier):
    """
    Compute the accuracy of a given a testset and using a map classifier object.
    
    Input
        - testset: The test for which to compute the accuracy (Numpy array).
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / #testset size
    """
    correct = 0
    tets_data = testset.iloc[:,:-1]
    for i,j in testset.iterrows():
        result = j.iloc[-1]
        data = j.iloc[:-1]
        prediction = map_classifier.predict(data)
        if result == prediction:
            correct += 1
    return correct / tets_data.shape[0]
    
            
            
            
            
            
            
            
            
            
    