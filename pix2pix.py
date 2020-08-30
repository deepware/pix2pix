import os, sys, glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

device = 'cuda:0'


class ConvEnc(nn.Module):
	def __init__(self, in_chan, out_chan, normalize=True, stride=2):
		super(ConvEnc, self).__init__()

		conv = nn.Conv2d(in_chan, out_chan, 4, stride=stride, padding=1)
		nn.init.normal_(conv.weight, 0, 0.02)
		model = [conv]

		if normalize:
			norm = nn.InstanceNorm2d(out_chan)
			model += [norm]

		model += [nn.LeakyReLU(0.2)]
		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)


class ConvDec(nn.Module):
	def __init__(self, in_chan, out_chan, dropout=False):
		super(ConvDec, self).__init__()

		conv = nn.ConvTranspose2d(in_chan, out_chan, 4, stride=2, padding=1)
		nn.init.normal_(conv.weight, 0, 0.02)
		model = [conv]

		norm = nn.InstanceNorm2d(out_chan)
		model += [norm]

		if dropout:
			model += [nn.Dropout(0.5)]

		model += [nn.ReLU()]
		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.enc1 = ConvEnc(3, 64, normalize=False)
		self.enc2 = ConvEnc(64, 128)
		self.enc3 = ConvEnc(128, 256)
		self.enc4 = ConvEnc(256, 512)
		self.enc5 = ConvEnc(512, 512)
		self.enc6 = ConvEnc(512, 512)
		self.enc7 = ConvEnc(512, 512)
		self.bott = ConvEnc(512, 512, normalize=False)

		self.dec7 = ConvDec(512, 512, dropout=True)
		self.dec6 = ConvDec(1024, 512, dropout=True)
		self.dec5 = ConvDec(1024, 512, dropout=True)
		self.dec4 = ConvDec(1024, 512)
		self.dec3 = ConvDec(1024, 256)
		self.dec2 = ConvDec(512, 128)
		self.dec1 = ConvDec(256, 64)

		self.conv = nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1)

	def forward(self, x):
		enc1 = self.enc1(x)
		enc2 = self.enc2(enc1)
		enc3 = self.enc3(enc2)
		enc4 = self.enc4(enc3)
		enc5 = self.enc5(enc4)
		enc6 = self.enc6(enc5)
		enc7 = self.enc7(enc6)

		bott = self.bott(enc7)

		dec7 = self.dec7(bott)
		dec6 = self.dec6(torch.cat((dec7, enc7), 1))
		dec5 = self.dec5(torch.cat((dec6, enc6), 1))
		dec4 = self.dec4(torch.cat((dec5, enc5), 1))
		dec3 = self.dec3(torch.cat((dec4, enc4), 1))
		dec2 = self.dec2(torch.cat((dec3, enc3), 1))
		dec1 = self.dec1(torch.cat((dec2, enc2), 1))
		conv = self.conv(torch.cat((dec1, enc1), 1))

		return torch.tanh(conv)


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = ConvEnc(6, 64, normalize=False)
		self.conv2 = ConvEnc(64, 128)
		self.conv3 = ConvEnc(128, 256)
		self.conv4 = ConvEnc(256, 512, stride=1)
		self.conv5 = nn.Conv2d(512, 1, 4, padding=1)

	def forward(self, a, b):
		x = torch.cat((a, b), 1)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.conv5(x)
		return x


def train(G, D, optim_G, optim_D, train_loader, val_loader, epoch):
	G.train()
	D.train()

	G_loss, D_loss = [], []
	gan_loss = F.binary_cross_entropy_with_logits

	for i, (a, b) in enumerate(train_loader):

		a = a.to(device)
		b = b.to(device)

		real = torch.ones((a.size(0), 1, 30, 30)).to(device)
		fake = torch.zeros((a.size(0), 1, 30, 30)).to(device)

		# update generator

		optim_G.zero_grad()

		fake_b = G(a)
		pred = D(a, fake_b)

		loss_G = gan_loss(pred, real) + F.l1_loss(b, fake_b) * 100

		loss_G.backward()
		optim_G.step()

		# update discriminator

		optim_D.zero_grad()

		pred_real = D(a, b)
		loss_real = gan_loss(pred_real, real)

		pred_fake = D(a, fake_b.detach())
		loss_fake = gan_loss(pred_fake, fake)

		loss_D = (loss_real + loss_fake) * 0.5

		loss_D.backward()
		optim_D.step()

		G_loss.append(loss_G.item())
		D_loss.append(loss_D.item())

		if (i+1) % 10 == 0:
			g_mean = np.mean(G_loss[-10:])
			d_mean = np.mean(D_loss[-10:])
			print("\rEpoch %d [%d/%d] [G loss: %f] [D loss: %f]" %
					(epoch, i, len(train_loader), g_mean, d_mean))

		iters = epoch * len(train_loader) + i
		if iters % 500 == 0:
			imgs = []
			for j, (a, b) in enumerate(val_loader):
				a = a.to(device)
				b = b.to(device)
				with torch.no_grad():
					fake_b = G(a)
				imgs += [a[0], fake_b[0], b[0]]
				if j == 2:
					break
			imgs = torch.stack(imgs).detach().cpu()
			save_image(imgs, "images/%s.png"%iters, nrow=3, normalize=True)

	return np.mean(G_loss), np.mean(D_loss)


def load_image(file):
	img = Image.open(file)
	img_b = img.crop((0,0,256,256))
	img_a = img.crop((256,0,512,256))
	T = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5),
		                     (0.5, 0.5, 0.5)),
	])
	return T(img_a), T(img_b)


class dataset(Dataset):
	def __init__(self, root, mode):
		self.files = glob.glob('%s/%s/*'%(root, mode))
		self.mode = mode

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		a, b = load_image(self.files[idx])
		return a, b


def main():

	if len(sys.argv) != 2:
		print('usage: pix2pix.py <dataset>')
		return 1

	root = sys.argv[1]
	os.makedirs("images/", exist_ok=True)

	G = Generator().to(device)
	D = Discriminator().to(device)

	epoch0 = -1
	if os.path.exists('model.pt'):
		state = torch.load('model.pt', map_location='cpu')
		G.load_state_dict(state['G'])
		D.load_state_dict(state['D'])
		epoch0 = state['epoch']

	optim_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optim_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

	train_loader = DataLoader(dataset(root, "train"), 1, shuffle=True)
	val_loader = DataLoader(dataset(root, "val"), 1, shuffle=True)

	for epoch in range(epoch0+1, 100):
		G_loss, D_loss = train(G, D, optim_G, optim_D, train_loader, val_loader, epoch)

		print('\nG_loss: %.4f	D_loss: %.4f\n'%(G_loss, D_loss))

		with open('log.txt', 'a') as f:
			print('%d\t%.4f\t%.4f'%(epoch, G_loss, D_loss), file=f)

		torch.save({
			'epoch': epoch,
			'G': G.state_dict(),
			'D': D.state_dict(),
			}, "model.pt")


if __name__ == "__main__":
	torch.backends.cudnn.benchmark = True
	main()
