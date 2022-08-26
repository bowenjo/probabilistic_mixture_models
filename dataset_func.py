import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from context import rf_pool
from PIL import Image
from rf_pool.data.stimuli import make_crowded_stimuli
import pickle

from scipy.stats import truncnorm



def make_circle(radius, image_size):
    c = int(image_size/2)
    xx, yy = np.mgrid[:image_size, :image_size]
    circle = (xx - c)**2 + (yy - c)**2
    return (circle < radius**2).astype(int)

def make_gaussian(sigma, image_size):
    c = int(image_size/2)
    xx, yy = np.mgrid[:image_size, :image_size]
    a = 1./ (sigma * np.sqrt(2.*np.pi))
    G = a * np.exp(-((xx - c)**2. + (yy - c)**2.) / (2.*sigma**2.))
    G = (np.max(G) - G) / (np.max(G) - np.min(G))
    return 1-G

def make_grating(orientation, diameter, n_cycles, contrast):
    assert (contrast <= 1 and contrast >= 0), (
        "Contrast must be a value between 0 and 1")
    orientation = (np.pi/2) - orientation
    phase = 0#np.random.uniform(0., 2*np.pi)
    xx, yy = np.meshgrid(np.arange(diameter), np.arange(diameter))
    
    circle_mask = make_circle(diameter/2, diameter)
    G = 1#make_gaussian(diameter/6, diameter)
    radians_per_cycle = (2*np.pi)/(diameter/n_cycles)
    shift_x = np.cos(orientation) * radians_per_cycle
    shift_y = np.sin(orientation) * radians_per_cycle
    grating =  .5 * ((contrast * np.sin(shift_x*xx + shift_y*yy+phase) * circle_mask * G) + 1)
    return np.uint8(255*grating)

def gabor_filter(theta, filter_shape, wavelength, gamma, sigma=3):
    """
    Create gabor filter
    """
    # get x, y coordinates for filter (centered)
    psi = np.random.uniform(0., 2*np.pi)
    theta = (np.pi/2) - theta
    x = np.arange(filter_shape) - filter_shape // 2
    y = np.arange(filter_shape) - filter_shape // 2
    x, y = np.stack(np.meshgrid(x, y), axis=0)
    # update based on orientation
    x_0 = x * np.cos(theta) + y * np.sin(theta)
    y_0 = -x * np.sin(theta) + y * np.cos(theta)
    # create weight for filter
    weight = np.multiply(np.exp(-(x_0**2 + gamma**2 * y_0**2)/(2. * sigma**2)),
                       np.cos(2. * np.pi * x_0 / wavelength + psi))
    weight = (np.max(weight) - weight) / (np.max(weight) - np.min(weight))
    return np.uint8(255*weight)

def make_crowded_gratings(target_orientation, flanker_orientations, diameter, n_cycles,
                          contrast, **kwargs):
    args = [diameter, n_cycles, contrast]
    target = gabor_filter(target_orientation, *args)
    
    flankers = []
    for f_orientation in flanker_orientations:
        flankers.append(gabor_filter(f_orientation, *args))               

    return make_crowded_stimuli(target, flankers, **kwargs) 

def pick_ensemble_flankers(n_flankers, target, mu, sigma, min_angle, max_angle):
    
    distances = np.abs(np.array([min_angle, max_angle]) - mu)
    min_distance = np.min(distances)
    
    # get the flanker(s) to keep the mean given the target
    x, n = get_flanker(target, mu, min_distance)
    flanker_orientations = [x,]*n
    
    # set the remaining flankers
    while len(flanker_orientations) < n_flankers:
        # truncate the distribution
        d = np.minimum(distances/min_distance, n_flankers-(len(flanker_orientations)+1));
        lower_angle = mu - d[0]*min_distance;
        upper_angle = mu + d[1]*min_distance;
        # sample an orientation from the distribution
        if lower_angle != upper_angle:
            low, up = get_truncation(lower_angle, upper_angle, mu, sigma)
            f = truncnorm.rvs(low, up, loc=mu, scale=sigma);
        else:
            f = mu;     
        # get the min number of orientation(s) to keep the mean
        x,n = get_flanker(f, mu, min_distance);
        flanker_orientations += [f] + [x,]*n;
    
    return np.random.permutation(flanker_orientations);

def get_truncation(low, up, mu, sigma):
    return (low - mu) / sigma, (up - mu) / sigma

def get_flanker(f, mu, min_distance):
    # get n: min number of flankers to keep mu
    n = int(np.ceil(np.abs(mu-f)/min_distance));
    # get x: the orientation value to keep mu
    if n != 0:
        x = (mu*(1+n)-f)/n;
    else:
        x=np.nan;
    return x,n

class CrowdedGratings(torch.utils.data.Dataset):
    """
    Class for constructing crowded gratings
    
    Attributes
    ----------
    n - int - dataset size
    n_flankers - int - number of flankers
    target_orientations - tuple - orientation of the target gratings in radians
    flanker_dist - tuple - std dev of the flanker distribution away from the target orientaiton
    diameter - float - size of the gratings in pixels
    n_cylces - float - number of cycles/grating
    contrast - float (0,1) - contrast of the dark and light parts of the grating
    task_type - str - target or ensemble
    
    """
    def __init__(self, root_data=None, n=1, n_flankers=2, target_orientations=(np.pi/4,3*np.pi/4),
                 flanker_dist=(np.pi/4,np.pi/4), diameter=20., n_cycles=5.,
                 contrast=1., task_type="target", transform=None, **kwargs):
        
        super().__init__()
        self.diameter = diameter
        self.n_cycles = n_cycles
        self.contrast = contrast
        self.task_type = task_type
        
        self.transform = transform
        self.data = []
        self.labels = []
        
        if root_data:
            self.labels, self.root_ensemble_val = root_data 
            self.root_targets = self.root_ensemble_val[0,:]
            self.root_flankers = self.root_ensemble_val[1:,:]
            self.n = len(self.labels)
            self.make_stimuli(sample=False,**kwargs)
          
        else:
            self.n = n
            self.n_flankers = n_flankers
            self.target_orientations = target_orientations
            self.flanker_dist = flanker_dist
            self.make_stimuli(sample=True,**kwargs)
            
    def sample_label(self, choices, n):
        randii = np.random.randint(low=0, high=len(choices), size=n)
        return randii
    
    def sample_flanker(self, target, n):
        sigma = self.flanker_dist[1]
        if target > np.pi/2.:
            mu = target - self.flanker_dist[0]
        else:
            mu = target + self.flanker_dist[0]
            
        if self.task_type == "target":     
            return target, pick_ensemble_flankers(self.n_flankers, target, mu, sigma, 0., np.pi)
        
        elif self.task_type == "ensemble":
            return mu, pick_ensemble_flankers(self.n_flankers, mu, target, sigma, 0., np.pi )
        
        else:
            raise ValueError()
            
    def make_stimuli(self, sample=True, **kwargs):
        for i in range(self.n):
            if sample:
                # smaple the target/flankers
                target_label = self.sample_label(self.target_orientations, 1)
                if self.n_flankers:
                    target, flankers = self.sample_flanker(self.target_orientations[target_label[0]],
                                                           self.n_flankers)
                else:
                    target = self.target_orientations[target_label[0]]
                    flankers = []
                    
                self.labels.append(target_label[0])
            else:
                target = self.root_targets[i]
                flankers = self.root_flankers[:,i]
                
            # make the crowded stimulus
            s = make_crowded_gratings(target, flankers, self.diameter,
                                      self.n_cycles, self.contrast, **kwargs)
            # append the data
            self.data.append(s)

        self.data = torch.tensor(self.data, dtype=torch.uint8)
        if sample:
            self.labels = torch.tensor(self.labels)
        else:
            self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index] 
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform:
            img = self.transform(img)
        return (img, label)
    
    def __len__(self):
        return len(self.labels)