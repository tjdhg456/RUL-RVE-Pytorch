import random
import torch
import torch.nn as nn

seed = 99
random.seed(seed)
torch.manual_seed(seed)


class Encoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=300, latent_dim=2):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 2, latent_dim)
        
        
    def forward(self, x):
        h, _ = self.lstm(x)
        feat = h[:, -1, :]
        
        # Sample mu and var
        mu = self.fc_mu(feat)
        var = self.fc_var(feat)
        return mu, var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(Decoder, self).__init__()
        self.regressor = nn.Sequential(*[nn.Linear(latent_dim, 200),
                                        nn.Tanh(),
                                        nn.Linear(200, 1)])
                                    
    def forward(self, z):
        out = self.regressor(z)
        return out
			
	
def reparameterize(mu, logvar):
	"""
	Reparameterization trick to sample from N(mu, var) from
	N(0,1).
	:param mu: (Tensor) Mean of the latent Gaussian [B x D]
	:param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
	:return: (Tensor) [B x D]
	"""
	std = torch.exp(0.5 * logvar)
	eps = torch.randn_like(std)
	return eps * std + mu  


if __name__=='__main__':
    device = 'cuda:1'
    input = torch.ones([128, 30, 5]).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)    
    
    mu, var = encoder(input)
    z = reparameterize(mu, var)
    out = decoder(z)
    