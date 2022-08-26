import numpy as np
import torch
from context import rf_pool


class BaseModel(torch.nn.Module):
    def __init__(self, k=.5):
        super().__init__()
        self.softmax = torch.nn.Softmax(1)
        self.k = k
    
    def set_data(self, X, resp, correct):
        self.correct = torch.tensor(correct, dtype=torch.float32)
        self.resp = torch.tensor(resp * (np.pi/180), dtype=torch.float32)
        self.X = torch.tensor(X * (np.pi/180), dtype=torch.float32)
        self.n = self.X.shape[0]
        
    def center_data(self, X, resp):
        centered = X + (np.pi/4)*torch.sign(np.pi/2-resp)
        return centered%np.pi
    
    def process_s_scale(self, s, scale):
        # make sure the scale is positive
        scale = torch.nn.functional.relu(scale)
        s = torch.nn.functional.relu(s)
        # make sure spatial weights sum to 1
        s_norm = torch.div(s,torch.sum(s))
        return s_norm, scale
        
    def RF_forward(self, X):
        # rf_model forward pass
        rf_output = self.softmax(self.model(X))
        # orientation contribution from each rf
        return torch.sum(rf_output * self.orientations, 1).T
        
    def raw_mean_likelihood(self, X, resp, s, scale):
        s, scale = self.process_s_scale(s,scale)
        # compute likelihood
        weighted_mean = torch.matmul(s,X)
        dist = np.pi/4 - torch.abs(weighted_mean - resp)
        return torch.sigmoid(scale*dist)
        
    def total_log_likelihood(self, p, model_likelihood):
        self.total_likelihood = p*self.k + (1-p)*model_likelihood
        return -torch.sum(torch.log(self.total_likelihood*.99+.005))

    
class RandomCorrect(BaseModel):
    """
    Correct response rate - no dependence on the ensemble
    
    Attributes:
    ----------
    p - float (0,1) - probability of getting an incorrect response
    """
    def __init__(self,p,**kwargs):
        super().__init__(**kwargs)
        # fitting parameters
        self.p = p
    
    def forward(self):
        p = torch.clamp(self.p,0,1)
        return self.total_log_likelihood(p,self.correct)
    
    
class WeightedSub(BaseModel):
    """
    A weighted linear combination of the ensemble objects
    
    Attributes:
    -----------
    p - torch.tensor(size=[1]) (0,1) - probability of getting an incorrect response
    s - torch.tensor(size=[1,n]) - weights for each object of the ensemble
    scale - torch.tensor(size=[1]) - scale of the orientation similarity signmoid
    
    """
    def __init__(self,p,s,scale,**kwargs):
        super().__init__(**kwargs)
        # fitting parameters
        self.p = p
        self.s = s
        self.scale = scale
        
    def sub_likelihood(self,X,resp,s,scale):
        # compute likelihood
        s, scale = self.process_s_scale(s,scale)
        X_centered = self.center_data(X,resp)
        dist = np.pi/4 - torch.abs(X_centered - np.pi/2)
        return torch.matmul(s, torch.sigmoid(scale*dist))
        
    def forward(self):
        p = torch.clamp(self.p,0,1)
        model_likelihood = self.sub_likelihood(self.X,self.resp,self.s,self.scale)
        return self.total_log_likelihood(p,model_likelihood)

    
class WeightedMult(BaseModel):
    """
    A weighted multiplicative combination of the ensemble objects
    
    Attributes:
    -----------
    see WeightedSub
    
    """
    def __init__(self,p,s,scale,**kwargs):
        super().__init__(**kwargs)
        # fitting parameters
        self.p = p
        self.s = s
        self.scale = scale
    
    def mult_likelihood(self,X,resp,s,scale):
        #compute likelihood
        s, scale = self.process_s_scale(s,scale)
        X_centered = self.center_data(X,resp)
        dist = np.pi/4 - torch.abs(X_centered - np.pi/2)
        prob = torch.sigmoid(scale*dist)
        mult_prob = torch.prod(torch.pow(prob,s.T),0)
        mult_prob_inv = torch.prod(torch.pow(1 - prob,s.T),0)
        return mult_prob / (mult_prob+mult_prob_inv)
    
    def forward(self):
        p = torch.clamp(self.p,0,1)
        model_likelihood = self.mult_likelihood(self.X, self.resp, self.s, self.scale)
        return self.total_log_likelihood(p,model_likelihood) 


class SimpleWeightedAverage(BaseModel):
    def __init__(self, p, s1, s2, scale,**kwargs):
        super().__init__(**kwargs)
        # fitting parameters
        self.p = p
        self.s1 = s1
        self.s2 = s2 
        self.scale = scale
        
    def get_s(self):
        return torch.cat((self.s1, self.s2.repeat(self.n-1,1)))
    
    def forward(self):
        p = torch.clamp(self.p,0,1)
        s = self.get_s()
        model_likelihood = self.raw_mean_likelihood(self.X, self.resp, s, self.scale)
        return self.total_log_likelihood(p,model_likelihood)


class WeightedAverage(BaseModel):
    """
    A weighted mean of the ensemble objects
    
    Attributes:
    -----------
    see WeightedSub
    
    """
    def __init__(self, p, s, scale,**kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.s = s
        self.scale = scale
        
    def forward(self):
        p = torch.clamp(self.p,0,1)
        model_likelihood = self.raw_mean_likelihood(self.X, self.resp, self.s, self.scale)
        return self.total_log_likelihood(p,model_likelihood)
    
    
class RFWeightedSub(WeightedSub):
    def __init__(self, orientations, model, p, s, scale, **kwargs):
        super().__init__(p,s,scale,**kwargs)
        self.orientations=orientations
        self.model=model
        
    def forward(self,X,resp):
        p = torch.clamp(self.p,0,1)
        model_likelihood = self.sub_likelihood(self.RF_forward(X),resp,self.s,self.scale)
        return self.total_log_likelihood(p, model_likelihood)   
    

class RFWeightedMult(WeightedMult):
    def __init__(self, orientations, model, p, s, scale, **kwargs):
        super().__init__(p,s,scale,**kwargs)
        self.orientations=orientations
        self.model=model
    
    def forward(self,X,resp):
        p = torch.clamp(self.p,0,1)
        model_likelihood = self.mult_likelihood(self.RF_forward(X),resp,self.s,self.scale)
        return self.total_log_likelihood(p, model_likelihood)   
    
    
class RFWeightedAverage(WeightedAverage):
    def __init__(self, orientations, model, p, s, scale, **kwargs):
        super().__init__(p,s,scale,**kwargs)
        self.orientations=orientations
        self.model=model
    
    def forward(self,X,resp):
        p = torch.clamp(self.p,0,1)
        model_likelihood = self.raw_mean_likelihood(self.RF_forward(X),resp,self.s,self.scale)
        return self.total_log_likelihood(p, model_likelihood)   
