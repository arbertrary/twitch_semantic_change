import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import os
import argparse

"""
https://pytorch.org/hub/pytorch_vision_vgg/
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#vgg
https://towardsdatascience.com/using-predefined-and-pretrained-cnns-in-pytorch-e3447cbe9e3c
"""
os.environ['TORCH_HOME'] = "/home/stud/bernstetter/.cache/torch"


def get_image_tensors(in_dir, model):
    tensor_dict = {}
    for image in os.listdir(in_dir):
        emotename = os.path.splitext(image)[0]
        filepath = os.path.join(in_dir, image)
        try:
            input_image = Image.open(filepath).convert("RGB")
        except OSError:
            continue

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        input_batch = input_batch.to('cpu')
        model.to('cpu')

        with torch.no_grad():
            output = model(input_batch)

        # Tensor of shape 128
        tensor_dict[emotename] = output[0]
    return tensor_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="generating an embedding")
    parser.add_argument("-e", "--emote_dirs", type=str, help="Path to the emote word embedding model")
    parser.add_argument("-o", "--out", type=str, help="Path where to store the tensors")
    args = vars(parser.parse_args())

    emote_dirs = args["emote_dirs"]
    # emote_dirs = "../data/testdata/testimages"
    out_path = args["out"]
    # out_path = "vgg_emote_tensors.pt"

    # model = models.squeezenet1_0()
    model = models.squeezenet1_1(pretrained=True)
    # model = models.vgg19(pretrained=True)

    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=(1, 1))
    )
    print(model)
    # model.classifier = nn.Sequential(
    #     nn.Linear(in_features=25088, out_features=128)
    # )
    # print(model)
    # Set final layer to output size 128 in accordance to our embedding sizes
    # model.classifier[6] = nn.Linear(in_features=4096, out_features=128)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    out_tensors = {}
    for emote_source in ["bttv", "ffz", "global"]:
        in_path = os.path.join(emote_dirs, emote_source)
        out_tensors.update(get_image_tensors(in_path, model))

    torch.save(out_tensors, out_path)

    # test = torch.load(out_path)
    # print(test["AngelThump"])
    # print(test["AngelThump"].size())
