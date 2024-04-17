import argparse
import os

import cv2
import keyboard

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to your BGR images folder relative to /dataF.")
parser.add_argument("-o", "--output", help="Path to output folder relative to /dataF.")
args = parser.parse_known_args()[0]

path = "dataF/" + args.path
outPath = ""
if args.output is None or args.output == args.path:
    print("Files will be overwritten. It can be reverted by running this program again.\nDo you want to overwrite files? y/[n]")
    k = keyboard.read_key().lower()
    if k == 'y':
        outPath = path
    else:
        print("Program terminated.")
        exit(0)
else:
    outPath = "dataF/" + args.output

if not os.path.isdir(outPath):
    os.mkdir(outPath)

i = 0
for _, _, images in os.walk(path):
    print(len(images), "images in folder.")
    print(int(len(images) / 2), "should be converted\n")
    for image in images:
        if image[-5] == 'c':
            imgPath = path + "/" + image
            outImg = outPath + "/" + image
            img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)
            cv2.imwrite(outImg, img)
            print(imgPath, "\t->\t", outImg)
            i = i + 1

print("Images converted:", i)
