#!/usr/bin/env python

import argparse
import cv2
import dlib
import imutils
import nudged
import numpy as np
import os
import sys

from imutils import face_utils


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
    parser.add_argument("-i", "--image-directory", required=True, help="path to input images")
    parser.add_argument("-o", "--output-directory", help="path to output directory")
    return parser


def get_face_detector_and_landmark_predictor(shape_predictor):
    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    return (detector, predictor)


def read_image(path_to_image):
    if not os.path.exists(path_to_image):
        raise ValueError("Cannot find path: %s" % path_to_image)
    # load the image
    image = cv2.imread(path_to_image)
    if image is None:
        raise ValueError("Error reading image: %s" % path_to_image)
    return image


def convert_image_to_resized_grayscale(image, width):
    # resize the image to have the desired width
    resized_image = imutils.resize(image, width=width)
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    return gray


def detect_face_landmarks(detector, predictor, image):
    width = 500
    grayscale_image = convert_image_to_resized_grayscale(image, width)
    resized_scale = float(image.shape[1]) / grayscale_image.shape[1]

    # detect faces in the grayscale image
    rects = detector(grayscale_image, 1)

    # if there is more (or less) than one face, throw an exception
    if len(rects) != 1:
        raise ValueError("Expected 1 face in image, but found: %s" % len(rects))

    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(grayscale_image, rects[0])
    shape = face_utils.shape_to_np(shape)

    # scale the coordinates back to their original resolution
    shape = shape.astype(float)
    shape *= resized_scale

    # put the shape locations into a dict
    face_dict = {}
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        face_dict[name] = shape[i:j]

    return face_dict


def process_image(detector, predictor, path_to_image):
    image = read_image(path_to_image)
    face_dict = detect_face_landmarks(detector, predictor, image)
    return {"path": path_to_image, "image": image, "face_dict": face_dict}


def process_images_in_directory(detector, predictor, path_to_directory):
    if not os.path.exists(path_to_directory):
        raise ValueError("Cannot find path to directory: %s" % path_to_directory)

    # Get the filenames in sorted order
    filenames = [os.path.join(path_to_directory, f) for f in os.listdir(path_to_directory) if os.path.isfile(os.path.join(path_to_directory, f))]
    filenames.sort()

    # Process the files in order
    processed_images = [process_image(detector, predictor, path_to_image) for path_to_image in filenames]

    return processed_images


def compute_affine_fit(from_pts, to_pts):
    trans = nudged.estimate(from_pts, to_pts)
    matrix = np.array(trans.get_matrix())
    return matrix[0:2, :]


def compute_affine_transforms(processed_images):
    def get_points(face_dict, keywords):
        points = []
        for keyword in keywords:
            points.extend(face_dict[keyword])
        return points

    keywords = ["mouth", "right_eyebrow", "left_eyebrow", "right_eye", "left_eye", "nose"]
    points = [get_points(processed_image["face_dict"], keywords) for processed_image in processed_images]

    transforms = [compute_affine_fit(points[i], points[0]) for i in range(1, len(points))]
    return transforms


def transform_images(processed_images, transforms):
    if len(processed_images) < 1:
        raise ValueError("Expected at least 1 image")
    if len(transforms) != len(processed_images) - 1:
        raise ValueError("Expected %s transforms but found %s" % (len(processed_images) - 1, len(transforms)))

    first_image = processed_images[0]["image"]
    dims = (first_image.shape[1], first_image.shape[0])

    transformed_images = [cv2.warpAffine(processed_images[i + 1]["image"], transforms[i], dims) for i in range(len(transforms))]
    transformed_images.insert(0, first_image)
    return transformed_images


def save_images(processed_images, transformed_images, path_to_output_directory):
    if len(processed_images) != len(transformed_images):
        raise ValueError("# input images: %s but # of transformed images: %s" % (len(processed_images), len(transformed_images)))
    if not os.path.exists(path_to_output_directory):
        os.makedirs(path_to_output_directory)

    for i in range(len(processed_images)):
        path = os.path.join(path_to_output_directory, os.path.basename(processed_images[i]["path"]))
        cv2.imwrite(path, transformed_images[i])

def main():
    parser = get_argument_parser()
    args = parser.parse_args()

    (detector, predictor) = get_face_detector_and_landmark_predictor(args.shape_predictor)

    processed_images = process_images_in_directory(detector, predictor, args.image_directory)
    transforms = compute_affine_transforms(processed_images)
    transformed_images = transform_images(processed_images, transforms)

    if args.output_directory:
        print "Saving images to %s" % args.output_directory
        save_images(processed_images, transformed_images, args.output_directory)
    else:
        for image in transformed_images:
             cv2.imshow('image', image)
             cv2.waitKey(0)

main()

