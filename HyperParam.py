
class hyperP():
    random_seed=3333
    epochs=200
    batch_size=256
    early_stop=5
    regularizer=0.05
     
    def __init__(self,model_name='model.h5',input_size=None,layers=[0],activation='relu',dropout_rate=[0],optimizer=None,loss='mse'):
        self.model_name=model_name
        self.input_size=input_size
        self.layers=layers
        self.activation=activation
        self.dropout_rate=dropout_rate
        self.optimizer=optimizer
        self.loss=loss
        pass
         
    def model_name(self,value):
        self.model_name=value
        pass
    
    def inputsize(self,value):
        self.input_size=value
        pass
    
    def layers(self,value):
        self.layers=value
        pass
    def activation(self,value):
        self.activation=value
        pass
    
    def dropout(self,value):
        self.dropout=value
        pass
    
    def optimizer(self,value):
        self.optimizer=value
        pass
    
    def loss(self,value):
        self.loss=value
        pass