import time
import os
from torch.autograd import Variable
import torch
import random
import numpy as np
import numpy
import networks
from my_args import  args
from scipy.misc import imread, imsave
from AverageMeter import  *
import shutil
import subprocess as sp
from PIL import Image

torch.backends.cudnn.benchmark = True # to speed up the

model = networks.__dict__[args.netName](    channel=args.channels,
                                    filter_size = args.filter_size ,
                                    timestep=args.time_step,
                                    training=False)

model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    pretrained_dict = torch.load(args.SAVED_MODEL)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval() # deploy mode

save_which=args.save_which
dtype = args.dtype

tot_timer = AverageMeter()
proc_timer = AverageMeter()
end = time.time()

save_video = 'interpolated_raw.avi'
dump = open(os.devnull, 'w')
fps = '30'
crf = '18'
video = sp.Popen(['ffmpeg', '-framerate', fps, '-i', '-',
                  '-c:v', 'libx264', '-preset', 'veryslow', '-crf', crf, '-y',
                  save_video], stdin=sp.PIPE, stderr=dump)

dataset = '/local/scratch/pmh64/datasets/adobe240f'
scene = 'IMG_0028_binning_4x'
hr_dir = os.path.join(dataset, 'hr_frames', scene)
num_lr = len(os.listdir(hr_dir)) - 1
fps = 10

first = range(1, num_lr - fps, 120 // fps)
second = range(1 + 120 // fps, num_lr, 120 // fps)
for i1, i2 in zip(first, second):
    print(i1, i2)
    files = [os.path.join(hr_dir, f'frame_{i:05d}.png') for i in (i1, i2)]
    im1, im2 = [torch.from_numpy(np.transpose(imread(f), (2,0,1)) / 255.).type(dtype) for f in files]

    y_ = torch.FloatTensor()

    assert (im1.size(1) == im2.size(1))
    assert (im1.size(2) == im2.size(2))

    channel, intHeight, intWidth = im1.size()
    if not channel == 3:
        print(f'Skipping {i1}, {i2}')
        continue

    if intWidth != ((intWidth >> 7) << 7):
        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
        intPaddingLeft =int(( intWidth_pad - intWidth)/2)
        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
    else:
        intWidth_pad = intWidth
        intPaddingLeft = 32
        intPaddingRight= 32

    if intHeight != ((intHeight >> 7) << 7):
        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
        intPaddingTop = int((intHeight_pad - intHeight) / 2)
        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
    else:
        intHeight_pad = intHeight
        intPaddingTop = 32
        intPaddingBottom = 32

    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

    torch.set_grad_enabled(False)
    X0 = Variable(torch.unsqueeze(im1,0))
    X1 = Variable(torch.unsqueeze(im2,0))
    X0 = pader(X0)
    X1 = pader(X1)

    X0 = X0.cuda()
    X1 = X1.cuda()
    proc_end = time.time()
    y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
    y_ = y_s[save_which]

    proc_timer.update(time.time() -proc_end)
    tot_timer.update(time.time() - end)
    end  = time.time()
    print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
    X0 = X0.data.cpu().numpy()
    if not isinstance(y_, list):
        y_ = y_.data.cpu().numpy()
    else:
        y_ = [item.data.cpu().numpy() for item in y_]
    offset = [offset_i.data.cpu().numpy() for offset_i in offset]
    filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
    X1 = X1.data.cpu().numpy()


    X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
    y_ = [np.transpose(255.0 * item.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight,
                              intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for item in y_]
    offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
    filter = [np.transpose(
        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
        (1, 2, 0)) for filter_i in filter]  if filter is not None else None
    X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))

    timestep = args.time_step
    numFrames = int(1.0 / timestep) - 1

    X0 = Image.fromarray(np.uint8(X0)).convert('RGB')
    X0.save(video.stdin, 'PNG')
    for img in y_:
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img.save(video.stdin, 'PNG')

video.stdin.close()
video.communicate()
