from math import log
import matplotlib.pyplot as plt

def cus_log_loss(target, predicted):
    if len(predicted) != len(target):
        print("Data object initiated")
        return
	
    target = [float(x) for x in target] # converting target into float
    predicted = [min([max([x,1e-15]), 1-1e-15]) for x in predicted]
    plt.hist(predicted
         , bins = 10)
    plt.title("Predicted distribution", fontsize = 14)
    plt.show()
    
    
   
    return -1.0 / len(target) *  sum([ target[i] * log(predicted[i]) + (1.0 - target[i]) * log(1.0 - predicted[i]) for i in range(len(predicted))])