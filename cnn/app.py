from __future__ import print_function
import os
import argparse
import numpy as np
import subprocess
import cv2
import tensorflow as tf
import json
from CNN import CNN

# from eval import

FASA_EXEC = '/Users/Benze/Documents/NUS/multimedia computing/tianchiclassification/preprocess/FASA/FASA'
# ckpt_dir = os.path.abspath('/Users/Benze/Documents/NUS/multimedia computing/tianchiclassification/model/checkpoints')
truth_file = os.path.abspath('/Users/Benze/Documents/NUS/multimedia computing/tianchiclassification/test_app/truth.json')

PIXEL_DEPTH = 255.0

# ['505.npy', '52.npy', '284.npy', '155.npy', '48.npy', '368.npy', '160.npy', '111.npy', '228.npy']
# 111: bag
# 160: sweater
# 228: skirt
# 284: shoes
# 368: dress
# 48: boots
# 505: blouses
# 52: T-shirts
# 155: pants

LABLE_DICT = {
    0: 'blouses',
    1: 'T-shirts',
    2: 'shoes',
    3: 'pants',
    4: 'boots',
    5: 'dress',
    6: 'sweater',
    7: 'bag',
    8: 'skirt'
}


def load_hyper_parameters(hyper_param_file):
    # Define the model hyper parameters
    with open(hyper_param_file) as f:
        params = json.loads(f.read())
        return params


def load_truth(truth_file):
    truth_dict = dict()
    with open(truth_file) as f:
        labels = json.loads(f.read())
        for label in labels:
            img_list = labels[label]
            for img in img_list:
                truth_dict[img] = label
        return truth_dict


def parse_args():
    parser = argparse.ArgumentParser(description="recognize clothing images")
    parser.add_argument('input_folder')
    parser.add_argument('model_folder')
    parser.add_argument('output_folder')
    parser.add_argument('--image_size', default=32, type=int)
    args = parser.parse_args()
    return args


def read_bounding_box(path):
    with open(path) as f:
        line = f.readline()
        box = map(int, map(float, line.split(',')))
        return box


def process_image(image_file, box, image_size):
    origin_image = cv2.imread(image_file)
    # box_file_path = os.path.join(rect_folder, class_folder, image_id + '.txt')
    # box_img = origin_image.copy()
    # cv2.rectangle(box_img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0))
    # plt.imsave('box.png', box_img)
    crop_img = origin_image[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
    # plt.imsave('crop.png', crop_img)
    scaled_img = cv2.resize(crop_img, (image_size, image_size))
    # plt.imsave('scaled.png', scaled_img)
    gray_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('gray.png', gray_img)
    image_data = (gray_img.astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
    # plt.imsave('final.png', image_data)
    return image_data


def generate_data(input_folder, image_size):
    object_output_path = os.path.join(input_folder, os.pardir, 'box')
    print('box output folder {}'.format(object_output_path))
    if not os.path.exists(object_output_path):
        os.makedirs(object_output_path)
    subprocess.call([FASA_EXEC, '-i', '-p', input_folder, '-f', 'jpg', '-s', object_output_path])
    files = os.listdir(input_folder)
    num_of_images = len(files)
    data = np.ndarray(shape=(num_of_images, image_size, image_size), dtype=np.float32)
    images = list()
    boxes = list()
    for i, image_file in enumerate(files):
        box_file = os.path.join(object_output_path, image_file[:image_file.find('.')] + '.txt')
        object_box = read_bounding_box(box_file)
        image_data = process_image(os.path.join(input_folder, image_file), object_box, image_size)
        data[i, :, :] = image_data
        images.append(image_file)
        boxes.append(object_box)
    return data, images, boxes


def annotate_image(correct, image_file, box, true_label, predict_label, input_folder, output_folder):
    image = cv2.imread(os.path.join(input_folder, image_file))
    cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255))
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    if correct:
        cv2.putText(image, predict_label, (20, 30), font, 1, (0, 255, 0), 2)
    else:
        cv2.putText(image, predict_label + '({})'.format(true_label), (20, 30), font, 1, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(output_folder, image_file), image)


def main():
    args = parse_args()
    model_folder = os.path.abspath(args.model_folder)
    ckpt_dir = os.path.join(model_folder, 'checkpoints')
    hyper_params = load_hyper_parameters(os.path.join(ckpt_dir, 'hyper_params'))

    # generate data representation for images
    image_data, images, boxes = generate_data(args.input_folder, args.image_size)
    image_data = image_data.reshape(
                    (-1, args.image_size, args.image_size, 1)).astype(np.float32)

    # Create the default graph.
    graph = tf.Graph()
    with graph.as_default():
        # Set seed for repoducible results
        np.random.seed(10)

        # Return the container with data/labels for train/test datasets.
        # ctest = loader.run(category='Testing')
        batch_size = int(image_data.shape[0])
        sess = tf.Session()

        cnn = CNN(batch_size=batch_size,
                  image_size=args.image_size,
                  num_channels=hyper_params['num_channels'],
                  num_labels=hyper_params['num_labels'],
                  patch_size=hyper_params['patch_size'],
                  depth=hyper_params['depth'],
                  num_hidden=hyper_params['num_hidden'])
        saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

        with sess.as_default():
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                model_checkpoint_path = ckpt.model_checkpoint_path
                print(model_checkpoint_path)
                if not os.path.isabs(model_checkpoint_path):
                    model_checkpoint_path = os.path.join(ckpt_dir, model_checkpoint_path)
                saver.restore(sess, model_checkpoint_path)
            else:
                print("error loading checkpoint")

            # Evaluation section.
            # for batch in range(eval_batch):  # Number of batchs to process
            feed_dict = {cnn.train_data: image_data}
            batch_predictions = sess.run(cnn.predictions, feed_dict)

            truth_dict = load_truth(truth_file)
            num_of_truth = 0

            if not os.path.exists(args.output_folder):
                os.mkdir(args.output_folder)

            for img, prediction, box in zip(images, batch_predictions, boxes):
                predict_label = LABLE_DICT[prediction]
                if truth_dict[img] == predict_label:
                    annotate_image(True, img, box, truth_dict[img], predict_label, args.input_folder, args.output_folder)
                    num_of_truth += 1
                    print(img + ": " + predict_label + ' true')
                else:
                    annotate_image(False, img, box, truth_dict[img], predict_label, args.input_folder, args.output_folder)
                    print(img + ": " + predict_label + ' false')
            print("total number of true prediction: {}".format(num_of_truth))


if __name__ == '__main__':
    main()