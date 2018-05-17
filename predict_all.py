import argparse

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
from collections import OrderedDict
import numpy as np
from PIL import Image
import json
import os


def load_categories(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def create_classifier(hidden_units):
    layers = []
    total = len(hidden_units)
    if total < 2:
        raise ("Invalid number of input layers. Has to be more than 1")
    for idx, features in enumerate(hidden_units):
        if (idx + 1 == total):
            layers.append(('output', nn.LogSoftmax(dim=1)))
        else:
            name = 'fc' + str(idx + 1)
            layers.append((name, nn.Linear(features, hidden_units[idx + 1])))
            if (idx + 2 < total):
                relu_name = 'relu' + str(idx + 1)
                dropout_name = 'dropout' + str(idx + 1)
                layers.append((relu_name, nn.ReLU()))
                layers.append((dropout_name, nn.Dropout(p=0.5)))

    return nn.Sequential(OrderedDict(layers))


def load_checkpoint(checkpoint_file, gpu):
    if gpu:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file,
                                map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    hidden_units = checkpoint['hidden_units']
    classifier = create_classifier(hidden_units)
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(img):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    width, height = img.size
    if width > height:
        img.thumbnail((height, 256), Image.ANTIALIAS)
    else:
        img.thumbnail((256, width), Image.ANTIALIAS)
    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img = img.crop((
        half_the_width - 112,
        half_the_height - 112,
        half_the_width + 112,
        half_the_height + 112
    ))

    np_image = np.array(img)
    img = np_image / 255
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()
    img = Variable(img, volatile=True)
    return img


def predict(image_path, model, topk):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.eval()
    image = Image.open(image_path)
    image = process_image(image)
    output = model.forward(image)
    output = torch.exp(output).data
    probs, classes = output.topk(topk)
    ind = model.class_to_idx
    res = dict((v, k) for k, v in ind.items())
    classes = [res[x] for x in classes.cpu().numpy()[0]]
    return probs, classes


def get_result(inp, probs, classes, categories):

    return {
        'image': inp,
        'result': categories.get(classes[0], classes[0]),
        'confidence': str(probs[0][:].numpy()[0] * 100) + '%',
        'topk_classes': [categories.get(x, x) for x in classes],
        'topk_probs': [x for x in probs[0][:].numpy()]
    }


def job(checkpoint, inp):
    model = load_checkpoint(checkpoint, False)
    categories = load_categories('cat_to_name.json')
    probs, classes = predict(inp, model, 3)
    return get_result(inp, probs, classes, categories)


def predict_all(checkpoint):
    output = {}
    for subdir, dirs, files in os.walk('flowers/test'):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith('jpg'):
                print(filepath)
                output[file] = job(checkpoint, filepath)

    return output

if __name__ == '__main__':
    result = predict_all('checkpoint_vgg16.pth')
    print(result)
