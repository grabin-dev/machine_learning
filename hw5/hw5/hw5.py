import matplotlib.pyplot as plt
import pandas as pd
from numpy import concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
from sklearn.svm import SVC

SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    train_data = array([])
    train_labels = array([])
    test_data = array([])
    test_labels = array([])
    ###########################################################################
    # TODO: Implement the function                                            #
    ##########################################################################
    # Calculate test train amount
    train_amount = round(data.shape[0] * train_ratio)
    test_amount = data.shape[0] - train_amount

    # permutaions
    labeled_data = permutation(concatenate((data, labels.reshape(-1, 1)), axis=1))

    # Cutting to groups
    train_labeled_data = labeled_data[:train_amount,:].copy()
    test_labeld_data = labeled_data[train_amount:,:].copy()

    # Spliting to data and labels
    train_data = train_labeled_data[:,:-2]
    train_labels = train_labeled_data[:,-1]
    test_data = test_labeld_data[:,:-2]
    test_labels = test_labeld_data[:,-1]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param p)rediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the prediction
    """
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    tp = prediction[(labels == 1) & (prediction == 1)].shape[0]
    fp = prediction[(labels == 0) & (prediction == 1)].shape[0]
    tn = prediction[(labels == 0) & (prediction == 0)].shape[0]
    fn = prediction[(labels == 1) & (prediction == 0)].shape[0]

    tpr = tp / (fn + tp)
    fpr = fp / (tn + fp)
    accuracy = (tn + tp) / (tn + tp + fn + fp)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    if clf.gamma == 'auto_deprecated':
        clf.gamma = 'auto'
    for i in range(len(folds_array)):
        test_data = folds_array[i]
        test_label = labels_array[i]

        train_data = folds_array.copy()
        del train_data[i]
        train_data = concatenate(train_data)

        train_label = labels_array.copy()
        del train_label[i]
        train_label = concatenate(train_label)
        clf.fit(train_data,train_label)
        test_predict = clf.predict(test_data)
        tpr_k, fpr_k, accuracy_k = get_stats(test_predict,test_label)
        tpr.append(tpr_k)
        fpr.append(fpr_k)
        accuracy.append(accuracy_k)

    ####################################fgh#######################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    tpr = []
    fpr = []
    accuracy = []
    folds_array = array_split(ary=data_array,indices_or_sections=folds_count,axis=0)
    labels_array = array_split(labels_array,indices_or_sections=folds_count,axis=0)
    for kernel, param in zip(kernels_list,kernel_params):
        clf = SVC(kernel=kernel, **param)
        if clf.gamma == 'auto_deprecated':
            clf.gamma = SVM_DEFAULT_GAMMA
        tpr_k, fpr_k, accuracy_k = get_k_fold_stats(folds_array,labels_array,clf)
        tpr.append(tpr_k)
        fpr.append(fpr_k)
        accuracy.append(accuracy_k)

    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return svm_df


def get_most_accurate_kernel(res):
    """
    :param: table evaluate different kernels and params
    :return: integer representing the row number of the most accurate kernel
    """
    return res[max(res) ==  res].index[0]


def get_kernel_with_highest_score(res):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    return res[max(res) ==  res].index[0]


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    kernel = df.iloc[get_kernel_with_highest_score(df['score'])]
    tpr = kernel['tpr']
    fpr = kernel['fpr']
    b = tpr - (1.5 * fpr)
    line_x = array([0.01 * i for i in range(100)])
    line_y = alpha_slope * line_x + b
    plt.plot(line_x, line_y, label='y = {}x + b'.format(alpha_slope))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0.8, 1.1)
    plt.scatter(x, y)
    plt.legend()
    z = polyfit(x, y, 3)
    p = poly1d(z)
    import numpy as np
    xp = np.linspace(0, 1, 100)
    plt.plot(x, y, '.', xp, p(xp), '-')

    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count, kernel):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    # Calculate c values
    j_val = [1,2,3]
    i_val = [1,0,-1,-2,-3,-4]
    formula = lambda i,j : (10 ** i) * (j / 3)
    c_val = [formula(i,j) for i in i_val for j in j_val]
    c_params = [{'C':value} for value in c_val]

    #Compare params calculation
    kernel = [kernel for i in range(len(c_params))]
    res = compare_svms(data_array,labels_array,folds_count,kernel,c_params)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels,kernel ,kernel_parameters):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train
    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :param kernel_params: kernel params best choice
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = ''
    kernel_params = None
    clf = SVC(class_weight='balanced')  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    kernel_type = kernel
    kernel_params = kernel_parameters
    clf = SVC(kernel=kernel,**kernel_params)
    clf.fit(train_data,train_labels)
    prediction = clf.predict(test_data)
    tpr, fpr, accuracy = get_stats(prediction=prediction,labels=test_labels)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
