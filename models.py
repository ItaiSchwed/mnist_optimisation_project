import torch.nn as nn
import torch.nn.functional as F

# Hand-Written Numbers Mnist Model
class MnistModel(nn.Module):
	def __init__(self):
		super(MnistModel, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)

		return F.softmax(x, dim=1)

# Fashion Mnist Model
class FashionMnistModel(nn.Module):
	def __init__(self):
		super(FashionMnistModel, self).__init__()
		self.cnn1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.relu = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.cnn2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.fc = nn.Linear(7 * 7 * 32, 10)

	def forward(self, x):
		out = self.maxpool1(self.relu(self.bn1(self.cnn1(x))))
		out = self.maxpool2(self.relu(self.bn2(self.cnn2(out))))
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)

		return F.softmax(out, dim=1)