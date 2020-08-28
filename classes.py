# -*- coding: utf-8 -*-
# Created by Barron Pun

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

class GRU4Rec(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers=1, final_act='tanh', dropout_hidden=.5, dropout_input=0, batch_size=128, embedding_dim=64, use_cuda=False):
		super(GRU4Rec, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
		self.final_act = final_act
		self.dropout_hidden = dropout_hidden
		self.dropout_input = dropout_input
		self.batch_size = batch_size
		self.embedding_dim = embedding_dim
		self.use_cuda = use_cuda
		self.device = torch.device('cuda' if use_cuda else 'cpu')
		self.state = None



	def forward(self, inputs, state):
		pass


class Encoder(nn.Module):
	def __init__(self, rnn_size, hidden_size):
		super(Encoder, self).__init__()
		self.linear1 = nn.Linear(rnn_size, hidden_size)
		nn.init.xavier_normal_(self.linear1.weight)
		self.activation = nn.Tanh()

	def forward(self, x):
		x = self.linear1(x)
		x = self.activation(x)
		return x

class Decoder(nn.Module):
	def __init__(self, latent_size, hidden_size, num_items):
		super(Decoder, self).__init__()
		self.linear1 = nn.Linear(latent_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, num_items)
		nn.init.xavier_normal_(self.linear1.weight)
		nn.init.xavier_normal_(self.linear2.weight)
		self.activation = nn.Tanh()

	def forward(self, x):
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		return x


class SVAE(nn.Module):
	def __init__(self, rnn_size, hidden_size, latent_size, num_items, item_embed_size):
		super(SVAE, self).__init__()
		self.encoder = Encoder(rnn_size, hidden_size)
		self.decoder = Decoder(latent_size, hidden_size, num_items)
		
		self.item_embed = nn.Embedding(num_items, item_embed_size)
		
		self.gru = nn.GRU(item_embed_size, rnn_size, batch_first=True)
		
		self.linear1 = nn.Linear(hidden_size, 2*latent_size)
		nn.init.xavier_normal_(self.linear1.weight)
		
		self.tanh = nn.Tanh()

	def sample_latent(self, h_enc):
		"""
		Return the latent normal sample z ~ N(mu, sigma^2)
		"""
		temp_out = self.linear1(h_enc)
		
		mu = temp_out[:, :latent_size]
		log_sigma = temp_out[:, latent_size:]

		sigma = torch.exp(log_sigma)
		std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
		
		# std_z.to(device)
		
		self.z_mean = mu
		self.z_log_sigma = log_sigma

		return mu + sigma * Variable(std_z, requires_grad=False) # Reparameterization trick

	def forward(self, x):
		in_shape = x.shape
		x = x.view(-1)

		x = self.item_embed(x)
		x = x.view(in_shape[0], in_shape[1], -1)

		rnn_out, _ = self.gru(x)
		rnn_out = rnn_out.view(in_shape[0] * in_shape[1], -1)

		enc_out = self.encoder(rnn_out)
		sampled_z = self.sample_latent(enc_out)
		
		dec_out = self.decoder(sampled_z)
		dec_out = dec_out.view(in_shape[0], in_shape[1], -1) # (batch_size, sql, num_items)

		return dec_out, self.z_mean, self.z_log_sigma

class VAELoss(torch.nn.Module):
	def __init__(self, hyper_params):
		super(VAELoss,self).__init__()
		self.hyper_params = hyper_params

	def forward(self, decoder_output, mu_q, logvar_q, y_true_s, anneal):
		# Calculate KL Divergence loss
		kld = torch.mean(torch.sum(0.5 * (-logvar_q + torch.exp(logvar_q) + mu_q**2 - 1), -1))
	
		# Calculate Likelihood
		dec_shape = decoder_output.shape # [batch_size x seq_len x total_items] = [1 x seq_len x total_items]

		decoder_output = F.log_softmax(decoder_output, -1)
		num_ones = float(torch.sum(y_true_s[0, 0]))
		
		likelihood = torch.sum(
			-1.0 * y_true_s.view(dec_shape[0] * dec_shape[1], -1) * \
			decoder_output.view(dec_shape[0] * dec_shape[1], -1)
		) / (float(self.hyper_params['batch_size']) * num_ones)
		
		final = (anneal * kld) + (likelihood)
		
		return final


class DataReader:
	def __init__(self, hyper_params, a, b, num_items, is_training, device):
		self.hyper_params = hyper_params
		self.batch_size = hyper_params['batch_size']
		
		# num_users = 0
		# min_user = 1000000000000000000000000 # Infinity
		# for line in a:
		# 	line = line.strip().split(",")
		# 	num_users = max(num_users, int(line[0]))
		# 	min_user = min(min_user, int(line[0]))
		# num_users = num_users - min_user + 1
		
		self.num_users = len(a) # each line is a user
		# self.min_user = min_user
		self.num_items = num_items
		
		self.data_train = a
		self.data_test = b
		self.is_training = is_training
		self.device = device
		# self.all_users = []
		
		# self.prep()
		self.number()

	# def prep(self):
	# 	self.data = []
	# 	for i in range(self.num_users): self.data.append([])
			
	# 	for i in tqdm(range(len(self.data_train))):
	# 		line = self.data_train[i]
	# 		line = line.strip().split(",")
	# 		self.data[int(line[0]) - self.min_user].append([ int(line[1]), 1 ])
		
	# 	if self.is_training == False:
	# 		self.data_te = []
	# 		for i in range(self.num_users): self.data_te.append([])
				
	# 		for i in tqdm(range(len(self.data_test))):
	# 			line = self.data_test[i]
	# 			line = line.strip().split(",")
	# 			self.data_te[int(line[0]) - self.min_user].append([ int(line[1]), 1 ])
		
	def number(self):
		self.num_b = int(min(len(self.data_train), self.hyper_params['number_users_to_keep']) / self.batch_size)
	
	def iter(self):
		users_done = 0

		x_batch = []
		
		user_iterate_order = list(range(len(self.data_train)))
		
		# Randomly shuffle the training order
		np.random.shuffle(user_iterate_order)
		
		for user in user_iterate_order:

			if users_done > self.hyper_params['number_users_to_keep']: break
			users_done += 1
			
			y_batch_s = torch.zeros(self.batch_size, len(self.data_train[user]) - 1, self.num_items)
			# if is_cuda_available: y_batch_s = y_batch_s.cuda()
			y_batch_s.to(self.device)
			
			if self.hyper_params['loss_type'] == 'predict_next':
				for timestep in range(len(self.data_train[user]) - 1):
					y_batch_s[len(x_batch), timestep, :].scatter_(
						0, LongTensor([ i for i in [ self.data_train[user][timestep + 1] ] ]), 1.0
					)
				
			elif self.hyper_params['loss_type'] == 'next_k':
				for timestep in range(len(self.data_train[user]) - 1):
					y_batch_s[len(x_batch), timestep, :].scatter_(
						0, LongTensor([ i for i in self.data_train[user][timestep + 1:][:self.hyper_params['next_k']] ]), 1.0
					)
				
			elif self.hyper_params['loss_type'] == 'postfix':
				for timestep in range(len(self.data_train[user]) - 1):
					y_batch_s[len(x_batch), timestep, :].scatter_(
						0, LongTensor([ i for i in self.data_train[user][timestep + 1:] ]), 1.0
					)
			
			x_batch.append([ i for i in self.data_train[user][:-1] ])
			
			if len(x_batch) == self.batch_size: # batch_size always = 1
			
				yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False)
				x_batch = []

	def iter_eval(self):

		x_batch = []
		test_movies, test_movies_r = [], []
		
		users_done = 0
		
		for user in range(len(self.data_train)):
			
			users_done += 1
			if users_done > self.hyper_params['number_users_to_keep']: break
			
			if self.is_training == True: 
				split = float(self.hyper_params['history_split_test'][0])
				base_predictions_on = self.data_train[user][:int(split * len(self.data_train[user]))]
				heldout_movies = self.data_train[user][int(split * len(self.data_train[user])):]
			else:
				base_predictions_on = self.data_train[user]
				heldout_movies = self.data_test[user]
			
			y_batch_s = torch.zeros(self.batch_size, len(base_predictions_on) - 1, self.num_items).cuda()
			
			if self.hyper_params['loss_type'] == 'predict_next':
				for timestep in range(len(base_predictions_on) - 1):
					y_batch_s[len(x_batch), timestep, :].scatter_(
						0, LongTensor([ i for i in [ base_predictions_on[timestep + 1] ] ]), 1.0
					)
				
			elif self.hyper_params['loss_type'] == 'next_k':
				for timestep in range(len(base_predictions_on) - 1):
					y_batch_s[len(x_batch), timestep, :].scatter_(
						0, LongTensor([ i for i in base_predictions_on[timestep + 1:][:self.hyper_params['next_k']] ]), 1.0
					)
				
			elif self.hyper_params['loss_type'] == 'postfix':
				for timestep in range(len(base_predictions_on) - 1):
					y_batch_s[len(x_batch), timestep, :].scatter_(
						0, LongTensor([ i for i in base_predictions_on[timestep + 1:] ]), 1.0
					)
			
			test_movies.append([ i for i in heldout_movies ])
			test_movies_r.append([ i for i in heldout_movies ])
			x_batch.append([ i for i in base_predictions_on[:-1] ])
			
			if len(x_batch) == self.batch_size: # batch_size always = 1
				
				yield Variable(LongTensor(x_batch)), Variable(y_batch_s, requires_grad=False), test_movies, test_movies_r
				x_batch = []
				test_movies, test_movies_r = [], []














