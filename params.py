
import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--gpu_no',default="",type=str,help="using cpu if empty")
parser.add_argument('--is_train',action='store_true')
parser.add_argument('--model_name',choices=['rcan'])
parser.add_argument('--num_workers',type=int,default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--picture_format',default='png')


def make_params():
    params={}
    args,_=parser.parse_known_args()
    params['gpu_no']=args.gpu_no
    params['is_train']=args.is_train
    params['model_name']=args.model_name
    params['num_workers']=args.num_workers
    params['batch_size']=args.batch_size
    params['picture_format']=args.picture_format
    if args.model_name=='rcan':
        rcan_params={}
        parser.add_argument('--n_resgroups', type=int, default=4)
        parser.add_argument('--n_resblocks', type=int, default=5)
        parser.add_argument('--n_feats', type=int, default=64)
        parser.add_argument('--reduction', type=int, default=16)
        parser.add_argument('--scale', type=int, default=1)
        parser.add_argument('--rgb_range', type=int, default=3)
        parser.add_argument('--n_colors', type=int, default=3)
        parser.add_argument('--res_scale', type=int, default=1)
        args,_=parser.parse_known_args()
        rcan_params['n_resgroups'] = args.n_resgroups
        rcan_params['n_resblocks'] = args.n_resblocks
        rcan_params['n_feats']=args.n_feats
        rcan_params['reduction'] = args.reduction
        rcan_params['scale'] = args.scale
        rcan_params['rgb_range'] = args.rgb_range
        rcan_params['n_colors'] = args.n_colors
        rcan_params['res_scale'] = args.res_scale
        params['rcan_params']=rcan_params



    if args.is_train:
        train_params = {}
        parser.add_argument('--learning_rate', type=int, default=1e-4)
        parser.add_argument('--decay_step_size', type=int, default=10)
        parser.add_argument('--decay_rate', type=int, default=0.5)
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--picture_num',type=int)
        parser.add_argument('--dataset_dir', type=str)
        parser.add_argument('--patch_size',type=int)
        args,_=parser.parse_known_args()
        train_params['learning_rate'] = args.learning_rate
        train_params['decay_step_size'] = args.decay_step_size
        train_params['decay_rate'] = args.decay_rate
        train_params['epoch'] = args.epoch
        train_params['continue_train']=args.continue_train
        train_params['picture_num']=args.picture_num
        train_params['dataset_dir']=args.dataset_dir
        train_params['patch_size']=args.patch_size
        params['train_params']=train_params

    else:
        eval_params={}
        parser.add_argument('--video_path', type=str)
        parser.add_argument('--video_height', type=int, default=1e-4)
        parser.add_argument('--video_width', type=int, default=10)
        parser.add_argument('--patch_size', type=int, default=256)
        parser.add_argument('--stride', type=int, default=50)

        args,_=parser.parse_known_args()
        eval_params['video_path'] = args.video_path
        eval_params['video_height']=args.video_height
        eval_params['video_width']=args.video_width
        eval_params['patch_size']=args.patch_size
        eval_params['stride']=args.stride
        params['eval_params']=eval_params




    return params

