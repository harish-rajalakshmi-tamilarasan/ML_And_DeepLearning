import numpy as np

def softmax(model_out):
    # Compute the exponentials of the model outputs in a numerically stable way
    exp_model_out = np.exp(model_out - np.max(model_out, axis=0, keepdims=True))
    
    # Compute the sum of the exponentials
    sum_exp_model_out = np.sum(exp_model_out, axis=0, keepdims=True)
    
    # Normalize the exponentials to get the softmax probabilities
    softmax_model_out = exp_model_out / sum_exp_model_out
    print(softmax_model_out.shape)
    
    return softmax_model_out

# Example usage
model_out = np.array([[1.0, -12.0, 1223.0], 
                      [2.0, 12.6, -3.0], 
                      [5.0, 223.0, 33.0]])

print(np.max(model_out, axis=0, keepdims=True))

softmax_output = softmax(model_out)
print(np.sum(softmax_output,axis=0,keepdims=True))
print("Softmax output:\n", softmax_output)
