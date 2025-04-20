from PIL import Image
import os
import torch
import numpy as np

def save_images(img_tensors, img_names, save_dir, type="normal"):
    for img_tensor, img_name in zip(img_tensors, img_names):
        if type == "parse":
            tensor = img_tensor.clone().cpu()
        else:
            tensor = img_tensor.clone() * 255
            tensor = tensor.cpu().clamp(0,255)            
        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
            
        Image.fromarray(array).save(os.path.join(save_dir, img_name), quality=95)

def to_show(x):
    return torch.cat((x, x, x), axis=1)

def Parse_7_to_1(parse):
    # show parse
    b, c, h, w = parse.shape
    parse_show = parse.detach().cpu().numpy()
    parse_show = parse_show.reshape(b, c, -1).transpose(0,2,1)
    res = [np.argmax(item, axis=1) for item in parse_show]
    parse_show = np.array(res).reshape(b, h, w)
    parse_show = torch.from_numpy(parse_show.astype('uint8')).unsqueeze(1).cuda()
    return parse_show    # [0,6]