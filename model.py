# -*- coding: utf-8 -*-


from modelsummary import summary
import torch.nn.functional as F
import torch.nn as nn
import torch


### MODEL
##########################


class LeNet5(nn.Module):

	def __init__(self, num_classes, grayscale=False):
		super(LeNet5, self).__init__()

		self.grayscale = grayscale
		self.num_classes = num_classes

		if self.grayscale:
			in_channels = 1
		else:
			in_channels = 3

		self.features = nn.Sequential(

			nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
			nn.Tanh(),
			nn.MaxPool2d(kernel_size=2)
		)

		self.classifier = nn.Sequential(
			nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
			nn.Tanh(),
			nn.Linear(120 * in_channels, 84 * in_channels),
			nn.Tanh(),
			nn.Linear(84 * in_channels, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1)
		logits = self.classifier(x)
		probas = F.softmax(logits, dim=1)
		return logits, probas

if __name__ == '__main__':

	# Device
	device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
	model = LeNet5(10, True)
	model.to(device)
	summary(model, torch.ones(128, 1, 32, 32), batch_size=128, show_input=False)