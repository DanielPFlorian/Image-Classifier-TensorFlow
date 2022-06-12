import argparse
import json

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

import preprocess_image
import plot_probs as pp


def main():
    in_args = get_input_args()
    image = in_args.image_path
    top_k = in_args.top_k
    lab_to_cat = in_args.category_names
    model = load_model(in_args.model_path)
    if top_k and lab_to_cat:
        top_ps, labels = predict(image, model, top_k)
        cat_names = load_cat_names(lab_to_cat)
        top_k_cat = [cat_names[f'{i+1}'] for i in labels]
        print_probs(top_ps, top_k_cat, top_k)
        pp.plot_probs(image, top_ps, top_k_cat, top_k)
    elif top_k:
        top_ps, labels = predict(image, model, top_k)
        print_probs(top_ps, labels+1, top_k)
        pp.plot_probs(image, top_ps, labels+1, top_k)
    elif lab_to_cat:
        cat_names = load_cat_names(lab_to_cat)
        top_ps, labels = predict(image, model, len(cat_names))
        top_k_cat = [cat_names[f'{i+1}'] for i in labels]
        for top_ps, labels, top_k_cat in zip(top_ps, labels, top_k_cat):
            print("\nCategory Name: {}\
                 \nLabel: {}\
                 \nProbability: {}".format(top_k_cat, labels+1, top_ps))
    else:
        top_ps, labels = predict(image, model)
        print_probs(top_ps, labels+1)
        pp.plot_probs(image, top_ps, labels+1)


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",
                        help="File path of image")
    parser.add_argument("model_path",
                        help="Name of trained tensorflow model")
    parser.add_argument("--top_k", dest="top_k", type=int,
                        help="Return the top K most likely classes")
    parser.add_argument("--category_names", dest="category_names",
                        help="Maps labels to flower names")
    return parser.parse_args()

def load_model(model_path):
    return tf.keras.models.load_model(model_path,
                                      custom_objects={'KerasLayer':hub.KerasLayer})

def load_cat_names(label_map):
    with open(label_map, 'r') as f:
        return json.load(f)

def predict(image_path, model, top_k = 1):
        image = Image.open(image_path)
        image = preprocess_image.process_image(image)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        top_probs, labels = tf.math.top_k(pred, k=top_k)
        return top_probs.numpy()[0], labels.numpy()[0]

def print_probs(probs, labels, top_k=1):
    print("\nTop {} predicted category name(s):".format(top_k))
    for prob, label in zip(probs, labels):
        print("\nThe probability of {} is {}".format(label, prob))
    return

if __name__ == "__main__":
    main()
