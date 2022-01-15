import  cupy as  cp

from csrc.activation import SoftMax
from csrc.layers.layer import Layer

# Cu sinapsa comparativa GPU 

from csrc.comp_syn import cp_comp

class C2FullyConnected(Layer):
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
        self.w = (cp.random.randn(self.size, in_dim) * cp.sqrt(2 / in_dim)).astype('float32')
        # S-a trecut la tip float32 pentru a putea apela operatorul cp_comp 
        self.b = cp.zeros((1, self.size)).astype('float32')

    def forward(self, a_prev, training):
        #print('Forma1: ',cp.shape(a_prev))
        #print('Forma1: ',cp.shape(self.w.T))
        z = cp_comp(a_prev, self.w.T) + self.b  # strat comparativ  
       
        

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
        dw = 1 / batch_size * cp.dot(dz.T, a_prev)
        
        '''
        # aici ar trebui inlocuit  dz.T = (clase,batch) * (batch, intrari)
        m1=cp.shape(dz.T)[0]
        n1=cp.shape(a_prev)[0]
        n2=cp.shape(a_prev)[1]
        dw=cp.zeros((m1,n2))
        for k in range(m1):
            dw[k,:]=cp.sum(dz.T[k,:] * a_prev.T, axis=1)    
            #dw[k,:]=0.5*cp.sum(cp.abs(dz.T[k,:]+a_prev.T)-cp.abs(dz.T[k,:]-a_prev.T),axis=1)
            #dw[k,:]=0.002*cp.sum(cp.sign(dz.T[k,:]+a_prev.T)+cp.sign(dz.T[k,:]-a_prev.T),axis=1)
        dw = 1 / batch_size * dw
        
        #print('Forma dz.T : ',cp.shape(dz.T))
        #print('Forma a_prev : ',cp.shape(a_prev))

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
        da_prev = cp.dot(dz, self.w)
        #print('Forma dz: ',cp.shape(dz))
        #print('Forma w: ',cp.shape(self.w))



        return da_prev, dw, db

    def update_params(self, dw, db):
        self.w -= dw
        self.b -= db

    def get_params(self):
        return self.w, self.b

    def get_output_dim(self):
        return self.size
