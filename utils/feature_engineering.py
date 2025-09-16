import numpy as np

def feature_engineer(df):
    
    required_cols = { 'h_bar',
                      'mass',
                      'time',
                      'sig_0',
                      'sig_0_2',
                      'sig_t' }
    
    missing = required_cols - set(df.columns)
    
    if missing:
        raise ValueError(f'Missing required Columns: {missing}')
    
    df['t_c'] =  (2 * df['mass'] * df['sig_0_2']) / df['h_bar']
    
    df['norm_time'] = df['time'] / df['t_c']
    
    df['spreading_factor'] = 1 + np.square(df['norm_time'])

    return df