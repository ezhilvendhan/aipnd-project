import argparse

import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
from collections import OrderedDict
import numpy as np
from PIL import Image
import json


def parse_args():
    """ Parses commandline arguments and assigns default values if needed"""
    parser = argparse.ArgumentParser(description="Ezhil's Image Classifier")
    parser.add_argument('inp', action="store")
    parser.add_argument('checkpoint', action="store")
    parser.add_argument('--top_k', action='store',
                        default='3', dest='top_k', type=int)
    parser.add_argument('--category_names', action='store',
                        default='cat_to_name.json', dest='category_names')
    parser.add_argument('--gpu',
                        action='store_true', dest='gpu')

    args = parser.parse_args()
    print()
    print("Starting to predict using these inputs")
    print("{0: <30}".format('Image:'), args.inp)
    print("{0: <30}".format('Checkpoint:'), args.checkpoint)
    print("{0: <30}".format('top_k:'), args.top_k)
    print("{0: <30}".format('category_names:'), args.category_names)
    print("{0: <30}".format('gpu:'), args.gpu)
    print()
    print()
    return args


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


def predict(image_path, model, topk, gpu):
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


def display_result(inp, probs, classes, categories):
    print("=" * 7)
    print("RESULTS")
    print("=" * 7)
    print()
    print("{0: <30}".format('Input:'), inp)
    print("{0: <30}".format('Prediction:'), \
          categories.get(classes[0], classes[0]))
    print("{0: <30}".format('Probability:'), "{:.3f}%".format(probs[0][0] * 100))
    print()
    print("***** topk classes and probabilities *****")
    print("{0: <5}".format("Rank"),
          "{0: <25}".format("Prediction"), "{0: <25}".format("Probability"))
    for i in range(len(classes)):
        print("{0: <5}".format(str(i + 1)),
              "{0: <25}".format( \
                  categories.get(classes[i], classes[i])), \
              "{0: <25}".format("{:.3f}%".format(probs[0][i] * 100)))
    print()


if __name__ == "__main__":
    args = parse_args()
    model = load_checkpoint(args.checkpoint, args.gpu)
    categories = load_categories(args.category_names)
    probs, classes = predict(args.inp, model, args.top_k, args.gpu)
    display_result(args.inp, probs, classes, categories)

