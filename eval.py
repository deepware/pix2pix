import os, sys, glob, shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pix2pix as P

device = 'cuda:0'


G = P.Generator().to(device)
ckpt = torch.load('model.pt', map_location=device)
G.load_state_dict(ckpt['G'])

preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def process(file, save_dir):
	base = os.path.basename(file)
	img = Image.open(file).convert('RGB')
	img = img.crop((256,0,512,256)) # B to A
	x = preprocess(img)
	x = x.to(device).unsqueeze(0)
	y = G(x)[0].cpu()
	save_image(y, '%s/%s'%(save_dir, base), normalize=True)

def main():

	files = glob.glob(sys.argv[1]+'/*')
	os.mkdir('result')

	for i, file in enumerate(files):
		process(file, 'result')
		print('%d/%d\r'%(i+1, len(files)), end='')

main()
