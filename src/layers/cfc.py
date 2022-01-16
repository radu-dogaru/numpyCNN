import numpy as np

from src.activation import SoftMax
from src.layers.layer import Layer


class CFullyConnected(Layer):
    """Densely connected layer (comparative).

    Attributes
    ----------
    size : int
        Number of neurons.
    activation : Activation
        Neurons' activation's function.
    is_softmax : bool
        Whether or not the activation is softmax.
    cache : dict
        Cache.
    w : numpy.ndarray
        Weights.
    b : numpy.ndarray
        Biases.
    """
    def __init__(self, size, activation):
        super().__init__()
        self.size = size
        self.activation = activation
        self.is_softmax = isinstance(self.activation, SoftMax)
        self.cache = {}
        self.w = None
        self.b = None

    def init(self, in_dim):
        # He initialization
        self.w = (np.random.randn(self.size, in_dim) * np.sqrt(2 / in_dim)).astype('float32')

        self.b = np.zeros((1, self.size)).astype('float32')

    def forward(self, a_prev, training):
        #print('Forma1: ',np.shape(a_prev))
        #print('Forma1: ',np.shape(self.w.T))
        #z = np.dot(a_prev, self.w.T) + self.b  # strat normal 
        # strat comparativ - de lucrat 
        aaa=a_prev
        bbb=self.w.T
        bs=np.shape(aaa)[0]
        nc=np.shape(bbb)[1]
        z=np.zeros((bs,nc))
        for k in range(nc):
          z[:,k]=0.5*np.sum(abs(aaa+bbb[:,k])-abs(aaa-bbb[:,k]), axis=1)
        
        z=z+self.b
        #------------------------------
        

        a = self.activation.f(z)

        if training:
            # Cache for backward pass
            self.cache.update({'a_prev': a_prev, 'z': z, 'a': a})

        return a

    def backward(self, da):
        a_prev, z, a = (self.cache[key] for key in ('a_prev', 'z', 'a'))
        batch_size = a_prev.shape[0]

        # ------- aici propagarea erorii da prin neliniaritatea functiei de activare 

        if self.is_softmax:
            # Get back y from the gradient wrt the cost of this layer's activations
            # That is get back y from - y/a = da
            y = da * (-a)

            dz = a - y
        else:
            dz = da * self.activation.df(z, cached_y=a)


        #---------- aici update weights si bias --------
        dw = 1 / batch_size * np.dot(dz.T, a_prev)
        
        '''
        # aici ar trebui inlocuit  dz.T = (clase,batch) * (batch, intrari)
        m1=np.shape(dz.T)[0]
        n1=np.shape(a_prev)[0]
        n2=np.shape(a_prev)[1]
        dw=np.zeros((m1,n2))
        for k in range(m1):
            #dw[k,:]=np.sum(dz.T[k,:] * a_prev.T, axis=1)    
            #dw[k,:]=0.5*np.sum(np.abs(dz.T[k,:]+a_prev.T)-np.abs(dz.T[k,:]-a_prev.T),axis=1)
            dw[k,:]=0.002*np.sum(np.sign(dz.T[k,:]+a_prev.T)+np.sign(dz.T[k,:]-a_prev.T),axis=1)
        dw = 1 / batch_size * dw
        
        #print('Forma dz.T : ',np.shape(dz.T))
        #print('Forma a_prev : ',np.shape(a_prev))

        # NOTA: antrenarea cu sign() functioneaza numai cu gamma=0.002
        # optimizer=grad_descent si eta 1..10 --> rezulta max 83% 
        # pe fully connected cu USPS 
        # Cu un strat suplimentar merge "rau"
        # Pentru train e rcmd. sa ramana vechile formule !!
        # sign() cu tanh() devine antrenarea mai lenta 
        #----------- R.D. 26 iul 2021 ----------------
        '''
        
        db = 1 / batch_size * dz.sum(axis=0, keepdims=True)
        
        #------------ aici propagarea inversa a erorii 
        da_prev = np.dot(dz, self.w)
        #print('Forma dz: ',np.shape(dz))
        #print('Forma w: ',np.shape(self.w))



        return da_prev, dw, db

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.size
