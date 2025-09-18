from utils.data_processing import Preprocessor
from utils.feature_engineering import feature_engineer
from models.QWaveModel import QWaveModel
from utils.prob_distribution import electron_prob_density
import torch
import pandas as pd



DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

X = {
       'h_bar' : [1.00000000e+00],
       'mass' : [1.00000000e+00],
       'time': [1.93852073e+00],
       'sig_0': [6.00942957e+00],
       'sig_0_2': [3.61132437e+01]
                                   }

X = pd.DataFrame(X)

preprocessor = Preprocessor('results/scales/scaler_X.pkl', 'results/scales/scaler_y.pkl')

X = feature_engineer(X)

X  = preprocessor.scale_X(X)

X = torch.from_numpy(X).float().to(DEVICE)

model = QWaveModel(input_size=X.shape[1]).to(DEVICE)

state = torch.load('results/models/WavePacketProp_model.pth', map_location = DEVICE)

model.load_state_dict(state)

model.eval()

y_pred = model(X)

y = preprocessor.descale_y(y_pred.detach().cpu().numpy())

print(y)

# plot and probability distribution 