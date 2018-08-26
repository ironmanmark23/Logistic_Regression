# Logistic_Regression
In statistics, the logistic model (or logit model) is a statistical model that is usually taken to apply to a binary dependent variable. In regression analysis, logistic regression or logit regression is estimating the parameters of a logistic model.
More formally, a logistic model is one where the log-odds of the probability of an event is a linear combination of independent or predictor variables. The two possible dependent variable values are often labelled as "0" and "1", which represent outcomes such as pass/fail, win/lose, alive/dead or healthy/sick. The binary logistic regression model can be generalized to more than two levels of the dependent variable: categorical outputs with more than two values are modelled by multinomial logistic regression, and if the multiple categories are ordered, by ordinal logistic regression, for example the proportional odds ordinal logistic model.
https://en.wikipedia.org/wiki/File:Exam_pass_logistic_curve.jpeg
 
 
 
 
 PYTHON IMPLEMENTATION:
 
 
 
    #def weightInitialization(n_features):
       w = np.zeros((1,n_features))
       b = 0
       return w,b
     def sigmoid_activation(result):
          final_result = 1/(1+np.exp(-result))
         return final_result

    def model_optimize(w, b, X, Y):
        m = X.shape[0]
    #
    #Prediction
    final_result = sigmoid_activation(np.dot(w,X.T)+b)
    Y_T = Y.T
    cost = (-1/m)*(np.sum((Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result)))))
    #
    
    #Gradient calculation
    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T))
    db = (1/m)*(np.sum(final_result-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost
    def model_predict(w, b, X, Y, learning_rate, no_iterations):
        costs = []
        for i in range(no_iterations):
        #
             grads, cost = model_optimize(w,b,X,Y)
        #
             dw = grads["dw"]
             db = grads["db"]
        #weight update
             w = w - (learning_rate * (dw.T))
             b = b - (learning_rate * db)
        #
        
        if (i % 100 == 0):
            costs.append(cost)
            #print("Cost after %i iteration is %f" %(i, cost))
    
       #final parameters
       coeff = {"w": w, "b": b}
       gradient = {"dw": dw, "db": db}
    
       return coeff, gradient, costs
    def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred
    #
