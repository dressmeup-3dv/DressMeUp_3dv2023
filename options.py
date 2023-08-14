import argparse


def get_args(parser):
    #model
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type = int, default= 1)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--gpu', default=[0],nargs="+", type=int)
    parser.add_argument('--seed', default = 123)
    parser.add_argument('--logger', type = str, default = None)
    
    #experiment
    parser.add_argument('--model_path', type = str, default = "checkpoints/model.ckpt")
    parser.add_argument('--mode', type = str, default = 'test')
    parser.add_argument('--device', type = str, default="cuda:0")
    parser.add_argument('--real_scan', action='store_true')
    parser.add_argument('--save_texture', action='store_false')
    parser.add_argument('--bottomwear', action='store_true')


    #wandb 
    parser.add_argument('--project_name', default='demo')
    parser.add_argument('--experiment', type = str, default = 'demo')
    parser.add_argument('--garment_path', type = str, default="dataset/garment")
    parser.add_argument('--garment_name', type = str, default="")
    parser.add_argument('--wandb_id', type = str, default="")
    parser.add_argument('--comment', type = str, default="")

    
    
    # dataset = 'cloth_3d'
    parser.add_argument('--dataset',type=str, default="cloth3d")
    parser.add_argument('--smpl_path', type = str, default='dataset/target_meshes/')
    parser.add_argument('--smpl_emd', type = str, default='dataset/target_meshes/mesh/smpl_emd.npy')
    

    return parser


def get_parser():
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_parser()
    print(args)