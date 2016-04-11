from __future__ import print_function
import sys
import os
import subprocess

if len(sys.argv) != 3:
    print('preprocess.py <input_folder> <output_folder>')
    exit(0)

input_folder = sys.argv[1]
output_folder = sys.argv[2]
for image_class in os.listdir(input_folder):
    input_path = os.path.join(input_folder, image_class)
    output_path = os.path.join(output_folder, image_class)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # print(input_path, output_path)
    subprocess.call(['./FASA', '-i', '-p', input_path, '-f', 'jpg', '-s', output_path])
