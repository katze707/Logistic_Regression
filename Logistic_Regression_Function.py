import numpy as np
import pandas as pd
import statsmodels.api as sm
data = sm.datasets.fair.load_pandas().data
data['affairs'] = (data.affairs > 0).astype(int)
x = data.drop('affairs', axis = 1)
y = data['affairs']


# In[3]:


class logisticRegression():


    def __init__(self):
        self.coef = 0


    def sigmoid(self, array):
          return (1/(1+np.exp(-array)))


    def fit(self, x, y, alpha, threshold):

        """
        x: input pandas dataframe
        y: output pandas series (Class labels)

        step1: Check if the number of rows in x and y are the same. If not raise a value error with a message."""

        if len(x.index.tolist()) != len(y):
            raise ValueError('Rows are not equivalent in X and Y')

        """
        step2: Check if there is any missing value in the dataset (both for x and y).
               If there is, raise a value error with a message."""
        if x.columns.isnull().sum() != 0:
            raise ValueError ('X has missing value')
        if y.isnull().sum() != 0:
            raise ValueError ('Y has missing value')

        """step3: Check if there is any categorical value in x or y.
        If there is, raise a value error with a message."""
        if x.select_dtypes(include = ['object']).columns.tolist() != []:
            raise ValueError ('X has categorical value')
        if  y.dtypes == 'object' is True:
            raise ValueError('Y has categorical value')


        """
        step4: Transform both x and y into numpy.arrays (it is easier to work with arrays for matrix operations).
        """
        x = np.asarray(x)
        y = np.asarray(y) #[...,np.newaxis] #reshape y from (6366,) to (6366,1)

        """
        step5: Add bias to the input vector x. bias means add a column which is 1 across al the rows.
               This will increase the number of columns of x by 1. x.shape[1] will increase by 1.
        """
        ones = np.ones((x.shape[0],1))
        inputs = np.concatenate((ones, x), axis = 1)

        """
        step6: initialize self.coef.
               You can initialize the coef randomly.
               Use numpy.random.rand(size) or np.random.uniform(low=-1, high=1, size=(x.shape[1])).
               Think about the size of the coefficent array.
               Logically, you need to have a coefficient for each input variable as well as the bias. """

        self.coef = np.random.uniform(low = -1, high = 1, size = (inputs.shape[1]))

        """
        step7: create a list to save the cost values for each iteration.
        """

        cost_values = []

        """
        step8: while not converged and iteration number > 10000
                    calculate the predicted values
                    calculate the error
                    calculate the cost function and append it to the cost list
                    calculate the gradient in a way that gradient is
                                      gradient = (t(x) * (error))/(size_of_x) (number of rows)
                    adjust the coef in a way that
                                        coef = coef - alpha*gradient
                    adjust alpha in a way that
                                        alpha = alpha*0.95"""

        i=0
        notConverged = True
        while notConverged:

            coef_times_inputs = np.dot(inputs,self.coef)

            predicted_y = self.sigmoid(coef_times_inputs)

            error = predicted_y - y
            cost =(np.sum(y*np.log(predicted_y+0.001) + (1-y)*np.log(1-predicted_y+0.001))/x.shape[0])

            cost_values.append(cost)

            gradient = np.dot(inputs.T, error)/x.shape[0]

            self.coef = self.coef - (alpha*gradient)

            alpha = alpha*0.95

            i = i+1

            if (i>10000) and (np.mean(cost_values[-5:]) - cost_values[-1]) < threshold:
                notConverged = False


            """step 8: Check if the convergence criteria is satisfied:
                if you iterate at least as many times 10000
                if the difference between the [average of the last 5 cost values] and [the last cost value]
                is less than the threshold.

            You will not need to return anything because you are working on the coefs, which are class attributes
            """

    def predict_prob(self, x):

        """
        Convert x into numpy aray and add bias
        Check if size of self.coef is the same with the number of columns in x
        Using x and self.coef, make the predictions
        """
        x = np.asarray(x)
        ones = np.ones((x.shape[0],1))
        inputs = np.concatenate((ones, x), axis = 1)
        if inputs.shape[1] == self.coef.shape[0]:
            predictions = self.sigmoid(np.dot(inputs,self.coef))
        return predictions


    def predict_class(self, x):

        """
        Make discrete predictions. Instead of returning probabilities return 0 or 1.
        """
        predicted_y = self.predict_prob(x)
        predicted_y[predicted_y < 0.5] = 0
        predicted_y[predicted_y > 0.5] = 1
        return predicted_y

    def get_accuracy(self, x, y):

        """
        Calculate the accuracy rate
        number of true classification/total number of instances
        number of true classification is True positive + True negative
        """
        yhat = self.predict_class(x)

        True_Predictions = []
        for i in range(0,len(yhat)):
            if yhat[i] == y[i]:
                True_Predictions.append(yhat[i])

        ACC = float(len(True_Predictions))/float(len(y))
        print 'the Accuracy is: ',ACC

        False_Predictions = []
        for i in range(0,len(yhat)):
            if yhat[i] != y[i]:
                False_Predictions.append(yhat[i])

        TN = True_Predictions.count(0)
        TP = True_Predictions.count(1)
        FN = False_Predictions.count(0)
        FP = False_Predictions.count(1)
        print '# of True Negtive: ',TN
        print '# of True Postive: ',TP
        print '# of False Negtive: ',FN
        print '# of False Positive: ',FP

        TPR = float(TP)/float(TP+FN)
        print 'the True Positive Rate is: ', TPR
        FPR = float(1 - TPR)
        print 'the False Positive Rate is: ',FPR
        PPV = float(TP)/float(TP+FP+0.0001)
        print 'the Precision is: ', PPV
        F1 = float(2*TP)/float(2*TP+FP+FN+0.0001)
        print 'the F1 score is: ', F1
