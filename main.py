from osgeo import gdal
import math, random
import time
from PIL import Image
import numpy as np
import torch
import warnings
from osgeo import gdal
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
from metrics import psnr, ssim
from models.FFA import FFA
from models.decloudnet import DecloudNet
from models.PerceptualLoss import LossNetwork as PerLoss
import os

# from tensorboardX import SummaryWriter

warnings.filterwarnings('ignore')
from option import opt, model_name, log_dir
# from data_utils import *
from torchvision.models import vgg16

print('log_dir :', log_dir) # 'logs/'+model_name
print('model_name:', model_name) # its_train_decloudnet_3_9.pk model_name=opt.trainset+'_'+opt.net.split('.')[0]+'_'+str(opt.gps)+'_'+str(opt.blocks)

models_ = {
	'ffa': FFA(gps=opt.gps, blocks=opt.blocks),
	'decloudnet': DecloudNet(depths=(3,3,27,3,3),dims=((48, 96, 144, 96, 48)),
						  kernel_sizes=((3, 3, 3),
                                        (13, 13, 13),
                                        (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                                        (13, 13, 13),
                                        (3, 3, 3)),
										deploy=False, attempt_use_lk_impl=False),
}

start_time = time.time()
T = opt.steps


def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
	lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
	return lr


def train(net, loader_train, loader_test, optim, criterion):
	losses = []
	start_step = 0
	max_ssim = 0
	max_psnr = 0
	ssims = []
	psnrs = []
	if opt.resume and os.path.exists(opt.model_path):
		print(f'resume from {opt.model_path}')
		ckp = torch.load(opt.model_path)
		losses = ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step = ckp['step']
		max_ssim = ckp['max_ssim']
		max_psnr = ckp['max_psnr']
		psnrs = ckp['psnrs']
		ssims = ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else:
		print('train from scratch *** ')
		if not os.path.exists(opt.model_dir):
			os.mkdir(opt.model_dir)
	for step in range(start_step + 1, opt.steps + 1):
		net.train()
		lr = opt.lr
		if not opt.no_lr_sche:
			lr = lr_schedule_cosdecay(step, T)
			for param_group in optim.param_groups:
				param_group["lr"] = lr
		x, y = next(iter(loader_train))
		x_small = x[:,:,50:250,50:250]
		x = x.to(opt.device)
		x_small = x_small.to(opt.device)
		y = y.to(opt.device)
		out = net(x)
		out_cut = out[:,:,50:250,50:250]
		out_small = net(x_small)
		loss1 = criterion[0](out, y)
		loss3 = criterion[0](out_cut, out_small)
		if opt.perloss:
			loss2 = criterion[1](out, y)
			loss = loss1 + 0.04 * loss2
		
		loss = loss1 + 10*loss3
		loss.backward()

		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		print(
			f'\rtrain loss : {loss.item():.5f}, {loss1.item():.5f}, {loss3.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
			end='', flush=True)

		# with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
		#	writer.add_scalar('data/loss',loss,step)

		if step % opt.eval_step == 0:
			with torch.no_grad():
				ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)

			print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

			# with SummaryWriter(logdir=log_dir,comment=log_dir) as writer:
			# 	writer.add_scalar('data/ssim',ssim_eval,step)
			# 	writer.add_scalar('data/psnr',psnr_eval,step)
			# 	writer.add_scalars('group',{
			# 		'ssim':ssim_eval,
			# 		'psnr':psnr_eval,
			# 		'loss':loss
			# 	},step)
			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			if ssim_eval > max_ssim and psnr_eval > max_psnr:
				max_ssim = max(max_ssim, ssim_eval)
				max_psnr = max(max_psnr, psnr_eval)
				torch.save({
					'step': step,
					'max_psnr': max_psnr,
					'max_ssim': max_ssim,
					'ssims': ssims,
					'psnrs': psnrs,
					'losses': losses,
					'model': net.state_dict()
				}, opt.model_path)
				torch.save(net, os.path.join(opt.model_dir, 'Net_{}.pth'.format(step)))
				print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')

	np.save(os.path.join(opt.model_dir,f'numpy_files/{opt.net}_{opt.steps}_losses.npy', losses))
	np.save(os.path.join(opt.model_dir,f'numpy_files/{opt.net}_{opt.steps}_ssims.npy', ssims))
	np.save(os.path.join(opt.model_dir,f'numpy_files/{opt.net}_{opt.steps}_psnrs.npy', psnrs))
	torch.save(net, os.path.join(opt.model_dir, 'NetLast.pth'))


def test(net, loader_test, max_psnr, max_ssim, step):
	net.eval()
	torch.cuda.empty_cache()
	ssims = []
	psnrs = []
	# s=True
	for i, (inputs, targets) in enumerate(loader_test):
		inputs = inputs.to(opt.device);
		targets = targets.to(opt.device)
		pred = net(inputs)
		# # print(pred)
		# tfs.ToPILImage()(torch.squeeze(targets.cpu())).save('111.png')
		# vutils.save_image(targets.cpu(),'target.png')
		# vutils.save_image(pred.cpu(),'pred.png')
		ssim1 = ssim(pred, targets).item()
		psnr1 = psnr(pred, targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	# if (psnr1>max_psnr or ssim1 > max_ssim) and s :
	#		ts=vutils.make_grid([torch.squeeze(inputs.cpu()),torch.squeeze(targets.cpu()),torch.squeeze(pred.clamp(0,1).cpu())])
	#		vutils.save_image(ts,f'samples/{model_name}/{step}_{psnr1:.4}_{ssim1:.4}.png')
	#		s=False
	return np.mean(ssims), np.mean(psnrs)


class dataPreparation(Dataset):
	def __init__(self, trainPath, imgList,train):
		"""
		按照os.listdir返回结果迭代每次读入的图片，防止爆内存。

		:param trainPath: 训练数据路径（需要有云图像和真值图像在同一个父文件夹下）
		:param imgList: 图像列表，如果需要打乱需输入前自行打乱
		:param train: trainPath读入数据是否为训练数据，False表示测试数据(区别在于是否有清晰图像数据)
		"""
		super(dataPreparation, self).__init__()
		self.train=train
		self.imgList = imgList
		self.trainPath = trainPath
		self.cloudPath = os.path.join(self.trainPath, 'Cloud')
		self.clearPath = os.path.join(self.trainPath, 'Clear')
		self.cloudList = [os.path.join(self.cloudPath, x) for x in self.imgList]
		self.clearList = [os.path.join(self.clearPath, x) for x in self.imgList]

	def __getitem__(self, index):
		haze = Image.open(self.cloudList[index])
		clear = Image.open(self.clearList[index])
		haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
		return haze, clear

	def augData(self, data, target):
		if self.train:
			rand_hor = random.randint(0, 1)
			rand_rot = random.randint(0, 3)
			data = tfs.RandomHorizontalFlip(rand_hor)(data)
			target = tfs.RandomHorizontalFlip(rand_hor)(target)
			if rand_rot:
				data = FF.rotate(data, 90 * rand_rot)
				target = FF.rotate(target, 90 * rand_rot)
		data = tfs.ToTensor()(data)
		data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
		target = tfs.ToTensor()(target)
		return data, target

	def __len__(self):
		return len(self.imgList)


if __name__ == "__main__":
	rate = 0.8  # 训练集和验证集的比例
	dataPath = r'/home/server4/lkq/RS_Data/dataJPG/train'
	trainPath = os.path.join(dataPath, 'Cloud')
	imgList = os.listdir(trainPath)
	trainList = imgList[0:math.ceil(rate * len(imgList))]
	valList = imgList[math.ceil(rate * len(imgList)):]
	trainSetIter = DataLoader(dataset=dataPreparation(dataPath, trainList,True),
							  batch_size=2,
							  shuffle=True,
							  drop_last=False)
	valSetIter = DataLoader(dataset=dataPreparation(dataPath, valList,False),
							batch_size=2,
							shuffle=True)
	net = models_[opt.net]
	net = net.to(opt.device)
	if opt.device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	criterion = []
	criterion.append(nn.L1Loss().to(opt.device))
	if opt.perloss:
		vgg_model = vgg16(pretrained=True).features[:16]
		vgg_model = vgg_model.to(opt.device)
		for param in vgg_model.parameters():
			param.requires_grad = False
		criterion.append(PerLoss(vgg_model).to(opt.device))
	optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
						   eps=1e-08)
	optimizer.zero_grad()
	print(opt.net)
	train(net, trainSetIter, valSetIter, optimizer, criterion)
# torch.save(net, "Net_.pth")

# tifPath = r'/home/server4/lkq/RS_Data/OUT/ffa-tif'
# jpgPath = r'/home/server4/lkq/RS_Data/OUT/ffa-jpg'
# testPath = os.path.join(dataPath, 'Test')
# testAndOut(net,testPath,tifPath)
# Batch_Convert_tif_to_jpg(tifPath, jpgPath)
