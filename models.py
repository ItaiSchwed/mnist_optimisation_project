import torch.nn as nn
import torch.nn.functional as F
import torchvision

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

		return x

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

		return out


class FashionMnistModel2(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv2d_32 = nn.Conv2d(1, 32, 3, padding=1)
		self.conv2d_64 = nn.Conv2d(32, 64, 3, padding=1)
		self.max2d = nn.MaxPool2d(2, 2)
		self.conv2d_128 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv2d_256 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
		self.linear1 = nn.Linear(3 * 3 * 256, 256)
		self.linear2 = nn.Linear(256, 64)
		self.linear3 = nn.Linear(64, 10)
		self.batch2d1 = nn.BatchNorm2d(64)
		self.batch2d2 = nn.BatchNorm2d(256)
		self.batch1d = nn.BatchNorm1d(64)
		self.drop = nn.Dropout(p=0.3)
		self.flat = nn.Flatten()

	def forward(self, x):
		x = x.view(-1, 1, 28, 28)
		x = F.relu(self.conv2d_32(x))
		x = F.relu(self.conv2d_64(x))
		x = self.batch2d1(x)
		x = F.relu(self.max2d(x))
		x = self.drop(x)

		x = F.relu(self.conv2d_128(x))
		x = F.relu(self.conv2d_256(x))
		x = self.batch2d2(x)
		x = F.relu(self.max2d(x))
		x = self.drop(x)

		x = self.flat(x)
		x = F.relu(self.linear1(x))
		x = self.drop(x)
		x = F.relu(self.linear2(x))
		x = self.drop(x)
		x = self.batch1d(x)
		x = self.linear3(x)
		return x

CatsDogsModel = torchvision.models.densenet121(pretrained=True)

num_ftrs = CatsDogsModel.classifier.in_features
CatsDogsModel.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    nn.Linear(500, 2)
)