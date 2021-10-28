# Un exemplu util pentru a intelege constructia unor retele neuronale convolutionale (preluat din lpraat/numpyCNN
S-a adaugat sinapsa comparativa descrisa in https://www2.eecs.berkeley.edu/Pubs/TechRpts/1997/ERL-97-5.pdf 
https://www2.eecs.berkeley.edu/Pubs/TechRpts/1997/ERL-97-40.pdf
cu relevanta in implementari HW (in lucru, partea de propagare erori nu este rezolvata) 

Performanta (timp de antrenare / testare) nu este extraordinara (dat fiind ca NUMPY nu suporta GPU). 
Se poate incerca lucrul cu CUPY (metodele de baza sunt aceleasi de ex. np.dot() --> cp.dot() dar pentru a obtine accelerari 
variabilele trebuiesc stocate pe GPU 

Code is avalailable as the notebook numpy_cnn_comparative_v2.ipynb

<a href="https://colab.research.google.com/github/radu-dogaru/numpyCNN/blob/main/numpy_cnn_comparative_v2.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#-----------------------------------------------------------------------------------------------------------------
Original description from lpraat/numpyCNN 


# NumpyCNN
A simple vectorized implementation of a Convolutional Neural Network in plain numpy I wrote while learning about neural networks, aaaand more.   

##### Example

```python
# ... some imports here ...
mnist.init()
x_train, y_train, x_test, y_test = preprocess(*mnist.load())
    
cnn = NeuralNetwork(
    input_dim=(28, 28, 1),
    layers=[
        Conv(5, 1, 32, activation=relu),
        Pool(2, 2, 'max'),
        Dropout(0.75),
        Flatten(),
        FullyConnected(128, relu),
        Dropout(0.9),
        FullyConnected(10, softmax),
    ],
    cost_function=softmax_cross_entropy,
    optimizer=adam
)
    
cnn.train(x_train, y_train,
          mini_batch_size=256,
          learning_rate=0.001,
          num_epochs=30,
          validation_data=(x_test, y_test))
```


In mnist_cnn.py there is a complete example with a simple model I used to get 99.06% accuracy on the mnist test dataset.

## You can find an implementation of: 
#### Gradient Checking
To check the correctness of derivatives during backpropagation as explained [here](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)  
There are examples of its usage in the tests.

#### Layers
- FullyConnected (Dense)
- CFullyConnected (Dense with comparative synapse - added by R. Dogaru) 
- Conv (Conv2D)
- Cconv (with comparative synapse) - added by R. Dogaru 
- Pool (MaxPool2D, AveragePool2D)
- Dropout
- Flatten

#### Optimizers
- Gradient Descent
- RMSProp
- Adam
