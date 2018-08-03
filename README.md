# Selfie Align

This project helps to align selfies across multiple photos, using facial landmarks to resize and rotate the selfies.

The assumption is that the selfies are in portrait mode (the horizontal is shorter than the vertical), and that the selfies are ordered alphabetically.

The first selfie is used as the "base" that is used by the other selfies to be aligned to.

## Getting started

### Prerequisites
This project was only tested in Ubuntu 18.04.

### Installing

Clone the repository:
```
sudo apt-get install git
git clone https://github.com/AllisonAmazonite/selfie-align.git
```

Install dlib (https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/):
```
sudo apt-get install build-essential cmake libgtk-3-dev libboost-all-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install numpy scipy scikit-image dlib opencv-python imutils
```

Download a shape predictor
```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

### How to run
To align the selfies:
```
./align_selfies.py -p shape_predictor.dat -i selfie_input_directory -o selfie_output_directory
```

To create an animated gif:
```
cd output_directory
convert -resize 50% -delay 50 -loop 0 *.jpg selfies.gif
```

## References

This project was heavily reliant on the following tutorial: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/

Nudged library (used to rotate and scale the selfies): https://github.com/axelpale/nudged-py

OpenCV affine transforms: https://docs.opencv.org/2.4.13.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html



