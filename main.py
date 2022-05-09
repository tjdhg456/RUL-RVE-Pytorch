import utils
from model import Encoder, Decoder, reparameterize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import RUL_Dataset
from utils import EarlyStopping
from copy import deepcopy
import os
import argparse


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--save_dir', type=str, default='./checkpoint/FD001/base')
	parser.add_argument('--dataset', type=str, default='FD001')
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--device', type=str, default='cuda:0')
	args = parser.parse_args()
    
    
	# ------------------------------ DATA -----------------------------------
	dataset = args.dataset
	save_folder = args.save_dir
	os.makedirs(save_folder, exist_ok=True)

 	# sensors to work with: T30, T50, P30, PS30, phi
	sensors = ['s_3', 's_4', 's_7', 's_11', 's_12']
 	# windows length
	sequence_length = 30
	# smoothing intensity
	alpha = 0.1
	batch_size = 128
	# max RUL
	threshold = 125
	# Load Dataset
	x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset, sensors, sequence_length, alpha, threshold)
	tr_dataset = RUL_Dataset(x_train, y_train)
	val_dataset = RUL_Dataset(x_val, y_val)
	test_dataset = RUL_Dataset(x_test, y_test)
 
 	# Load Loader
	tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
	
 
	# --------------------------------------- MODEL ----------------------------------------
	timesteps = x_train.shape[1]
	input_dim = x_train.shape[2]
	intermediate_dim = 300
	latent_dim = 2
	epochs = 10000
	device = args.device
	lr = args.lr
	early_stopping_with_loss = False
  
	encoder = Encoder().to(device)
	decoder = Decoder().to(device)
 
 
	# ---------------------------- Optimizer and Early Stopping ----------------------------
	optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
	early = EarlyStopping(patience=10)

 
	# --------------------------------- Train and Validation --------------------------------
	for epoch in range(epochs):
		# Train
		encoder.train()
		decoder.train()
		
		tr_loss = 0.
		for tr_x, tr_y in tr_loader:
			tr_x, tr_y = tr_x.to(device), tr_y.to(device)
   
			optimizer.zero_grad()
   
			mu, var = encoder(tr_x)
			z = reparameterize(mu, var).float()
			out = decoder(z).view(-1)

			kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
			mse_loss = F.mse_loss(out, tr_y)
			loss = kl_loss + mse_loss
			loss.backward()
			optimizer.step()
   
			tr_loss += loss.item() / len(tr_loader)
			
  
		# Validation
		encoder.eval()
		decoder.eval()
		val_loss = 0.
		val_rmse = 0.
		for val_x, val_y in val_loader:
			val_x, val_y = val_x.to(device), val_y.to(device)
   
			with torch.no_grad():
				mu, var = encoder(val_x)
				z = reparameterize(mu, var)
				out = decoder(z).view(-1)

			kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
			rmse_loss = torch.sqrt(F.mse_loss(out, val_y) + 1e-6)
			loss = kl_loss + rmse_loss

			val_loss += loss / len(val_loader)
			val_rmse += rmse_loss.item() / len(val_loader)

		print('Epoch %d : tr_loss %.2f, val_loss %.2f, val_rmse %.2f' %(epoch, tr_loss, val_loss, val_rmse))
		param_dict = {'encoder': deepcopy(encoder.state_dict()), 'decoder': deepcopy(decoder.state_dict())}
  

		# Early Stopping
		if early_stopping_with_loss:
			early(val_loss, param_dict)
		else:
			early(val_rmse, param_dict)

		if early.early_stop == True:
			break
	

	# Save Best Model
	torch.save(early.model, os.path.join(save_folder, 'best_model.pt'))
 
 
	# --------------------------------- Test --------------------------------
	encoder.load_state_dict(early.model['encoder'])
	decoder.load_state_dict(early.model['decoder'])
 
	encoder.eval()
	decoder.eval()
 
	test_loss = 0.
	test_rmse = 0.
	for test_x, test_y in test_loader:
		test_x, test_y = test_x.to(device), test_y.to(device)

		with torch.no_grad():
			mu, var = encoder(test_x)
			z = reparameterize(mu, var)
			out = decoder(z).view(-1)

		kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim = 1), dim = 0)
		rmse_loss = torch.sqrt(F.mse_loss(out, test_y) + 1e-6)
		loss = kl_loss + rmse_loss

		test_loss += loss / len(test_loader)
		test_rmse += rmse_loss.item() / len(test_loader)

	print('Final Result : test loss %.2f, test_rmse %.2f' %(test_loss, test_rmse))
	with open(os.path.join(save_folder, 'result.txt'), 'w') as f:
		f.writelines('Final Result : test loss %.2f, test_rmse %.2f' %(test_loss, test_rmse))