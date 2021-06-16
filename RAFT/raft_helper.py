import argparse
import torch

from .core.raft import RAFT
from .core.utils.utils import InputPadder
import shlex

def load(model_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args(shlex.split('--model='+model_path))
    
    model = torch.nn.DataParallel(RAFT(args))
    # model = RAFT(args)
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.cuda()
    model.eval()
    return model

def get_flow(model, image1, image2):
    '''
    images: [1, 3, H, W] 0~255
    flows: [H, W, 2] normalized flows
    '''
    # image1 = image1.permute(2, 0, 1)
    # image2 = image2.permute(2, 0, 1)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1.cuda(), image2.cuda())
    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        # [2, H_pad, W_pad]
        flow_up = flow_up[0].cpu()/torch.Tensor(list(flow_up[0].shape[1:])).float().reshape(2, 1, 1)

    return padder.unpad(flow_up).permute(1, 2, 0)