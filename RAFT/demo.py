import sys
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from tqdm import tqdm


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, filename):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # print(img_flo.min(), img_flo.max())
    cv2.imwrite('result/'+filename, img_flo[:, :, ::-1].astype('uint8'))
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    # model = RAFT(args)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    images = glob.glob(os.path.join(args.path, '*.png')) + \
                glob.glob(os.path.join(args.path, '*.jpg'))
    
    images = sorted(images)
    flows = []
    with torch.no_grad():
        # images = np.random.permutation(images)
        for count, (imfile1, imfile2) in enumerate(tqdm(zip(images[:-1], images[1:]))):
            # print(imfile1.split('/')[-1], imfile2.split('/')[-1])
            # image1 = load_image(imfile1)
            image1 = load_image(images[0])
            image2 = load_image(imfile2)
            # print(image1.shape)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # print(image1.shape)
            # print(flow_low.shape, flow_up.shape)
            # print(flow_low.min(), flow_low.max())
            # print(flow_up.min(), flow_up.max())
            # flows.append(flow_up.cpu())
            viz((image1*0.25+image2*0.75), flow_up, f"{count+1}.jpg")
    
    # flow_current = torch.zeros_like(flows[0])
    # image1 = load_image(images[0])
    # for i in range(1, len(images)):
    #     # print(images[i])
    #     image2 = load_image(images[i])
    #     flow_current += flows[i-1]
    #     viz((image1*0.25+image2*0.75), flow_current, f"{i}.jpg")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # print(args)
    # assert False
    demo(args)
