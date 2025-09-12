import numpy as np
import pandas as pd

class QWavePacketSpread:
    
    def __init__(self):
        
        self.h_bar = 1
        self.mass = 1
    

    
    def data_gen(self, t, sig_0):
        
        h_bar = self.h_bar
        mass = self.mass
        
        sig_t = sig_0 * np.sqrt(
            
            1 + np.square(
                (h_bar * t) / (2 * mass * np.square(sig_0))
                )
            )
        
        return h_bar, mass, t, sig_0, np.square(sig_0), sig_t


def input_var_gen():
    
    sig_0 = np.random.uniform(0.1, 10)
    t  = np.random.uniform(0, 10)
    
    return sig_0, t


data = {
    
    'h_bar' : [],
    'mass'  : [],
    'time' : [],
    'sig_0' : [],
    'sig_0_2': [],
    'sig_t' : []
}


generator = QWavePacketSpread()


for i in range(50000):
    
    _sig_0, _t = input_var_gen()
    
    h_bar, mass, time, sig_0, sig_0_2, sig_t = generator.data_gen(_t, _sig_0)
    
    data['h_bar'].append(h_bar)
    data['mass'].append(mass)
    data['time'].append(time)
    data['sig_0'].append(sig_0)
    data['sig_0_2'].append(sig_0_2)
    data['sig_t'].append(sig_t)
    
    
df = pd.DataFrame(data)

df.to_csv('data/wave_packet_spread.csv', index=False)
        