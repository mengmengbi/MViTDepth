import torch
import datasets
import networks

def get_monovit_pretrained():
    depth_dict = torch.load(r"F:\rd\mobilevit_distilla_pose_nofreeze_light_decoder\models\1024_320\depth.pth")
    # new_dict = depth_dict
    new_dict = {}
    for k, v in depth_dict.items():
        name = k[7:]
        new_dict[name] = v

    depth = networks.DeepNet('mpvitnet')
    depth.load_state_dict({k: v for k, v in new_dict.items() if k in depth.state_dict()})
    # depth.load_state_dict({k: v for k, v in new_dict.items() if k in depth.state_dict()})

    return depth
