from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import os
from skimage import io
import glob
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy import ndimage as nd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from colorization import Colorizer
import argparse
from colorizers import *


app = Flask(__name__)
CORS(app)

# Function to colorize image using OpenCV
def colorize_image(image):
    # Your colorization logic using cv2 goes here
    # This is a dummy function, replace it with your actual implementation
    return [image] * 3  # Returning the same image multiple times for demonstration


#----------------------KNN Example-----------------
# This function extracts features based on gaussian (sigma = 3 and sigma = 7) and
# variance (size = 3)
def extract_all(img):

    img2 = img.reshape(-1)
    
    # First feature is grayvalue of pixel
    df = pd.DataFrame()
    df['GrayValue(I)'] = img2 

    # Second feature is GAUSSIAN filter with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    # Third feature is GAUSSIAN fiter with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3    

    # Third feature is generic filter that variance of pixel with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1
    
    return df

# This function extracts average pixel value of some neighbors
# frame size : (distance * 2) + 1 x (distance * 2) + 1
#default value of distance is 8 if the function is called without second parameter
def extract_neighbors_features(img,distance = 8):

    height,width = img.shape
    X = []

    for x in range(height):
        for y in range(width):
            neighbors = []
            for k in range(x-distance, x+distance +1 ):
                for p in range(y-distance, y+distance +1 ):
                    if x == k and p == y:
                        continue
                    elif ((k>0 and p>0 ) and (k<height and p<width)):
                        neighbors.append(img[k][p])
                    else:
                        neighbors.append(0)

            X.append(sum(neighbors) / len(neighbors))

    return X

# This function extracts superpixels
# Every cell has a value in superpixel frame so 
# It is extracting superpixel value of every pixel
def superpixel(image, status):    
    if status:
        segments = slic(img_as_float(image), n_segments=100, sigma=5, compactness=0.1, channel_axis=None)
    else:
        segments = slic(img_as_float(image), n_segments=100, sigma=5, compactness=0.1, channel_axis=None) 

    return segments



def calculate_mae(y_true,y_predict):
    
    # Calculate mean absolute error for every color according to MAE formula
    error_b = float(sum([abs(float(item_true) - float(item_predict)) for item_true, item_predict in zip(y_true[:,0], y_predict[:,0])]) / len(y_true))
    error_g = float(sum([abs(float(item_true) - float(item_predict)) for item_true, item_predict in zip(y_true[:,1], y_predict[:,1])]) / len(y_true))
    error_r = float(sum([abs(float(item_true) - float(item_predict)) for item_true, item_predict in zip(y_true[:,2], y_predict[:,2])]) / len(y_true))
    
    # Return aveage of colours error
    return (((error_b + error_g + error_r) / 3))


# Function to save predicted images in Outputs folder in Dataset folder
def save_picture(test_data,rgb_data_name,y_predict):
    
    # If Outputs folder is not exist in directory of Dataset create it
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    # Create an array for colorful image 
    height,width = test_data.shape
    data = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill the data with predicted RGB values
    tmp = 0
    for i in range(height):
        for k in range(width):
            data[i,k] = [y_predict[tmp][0], y_predict[tmp][1], y_predict[tmp][2]]
            tmp +=1
            
    # Save predicted image
    cv2.imwrite(f'outputs/{rgb_data_name}.jpg', data)
    return data

distance = 1
minImageIndex  = 1
maxImageIndex  = 50
#-------------------------------------------------------------------------


def measureMae(img1,img2):
    image1 = io.imread(img1)
    image2 = io.imread(img2)
    # image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    if image1.shape[-1] != image2.shape[-1]:
        image2 = image2[:, :, :image1.shape[-1]]
    difference = np.abs(image1 - image2)
    mae = np.mean(difference)
    print("MAE:", mae)
    return mae

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    image_file_ref = request.files['ref']
    image_file_cue = request.files['cue']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})


    # Save the uploaded image temporarily
    temp_filename = 'temp_image.jpg'
    ref_image = 'ref_image.jpg'
    cue_image = 'cue_image.jpg'
    image_file.save(temp_filename)
    image_file_ref.save(ref_image)
    image_file_cue.save(cue_image)
    #-----------------------------------------------------------------------
            #temp_filename = 'temp_image.jpg'
            #ref_image = 'ref_image.jpg'
            #cue_image = 'cue_image.jpg'

            
    #---------------------------------------------------------------
    # Load the image using cv2
    img = cv2.imread(temp_filename)
    
    # Perform colorization using cv2
    colorized = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    # Save the colorized image
    colorized_filename = 'colorized_image.jpg'
    cv2.imwrite(colorized_filename, colorized)

    #--------------------------------------------------------------------------
    #-----------------------------KNN------------------------------
    source_file = ref_image
    target_file = temp_filename
    groundtruth_file = ref_image
    # Read the single source, target, and groundtruth images
    source_image = cv2.imread(source_file)
    target_image = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)  # Assuming target image is grayscale
    groundtruth_image = cv2.imread(groundtruth_file)

        # Prepare y (b, g, r)
    y = source_image.reshape((-1, 3))

    # Prepare y_true (b, g, r)
    y_true = groundtruth_image.reshape((-1, 3))

    X1 = extract_all(target_image).values
    X2 = superpixel(target_image, False).reshape(-1, 1)
    X3 = extract_neighbors_features(target_image, distance)
    X = np.c_[X1, X2, X3]

    # Prepare X_test variable
    X1_test = extract_all(target_image).values
    X2_test = superpixel(target_image, False).reshape(-1, 1)
    X3_test = extract_neighbors_features(target_image, distance)
    X_test = np.c_[X1_test, X2_test, X3_test]

    # Train the model
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X, y)

    # Test the model
    y_predict = knn_clf.predict(X_test)

    # Calculate Mean Absolute Error
    # MAE = calculate_mae(y_true, y_predict)

    # Save Picture to Dataset/Outputs folder
    predicted_picture = save_picture(target_image, 'predicted_image', y_predict)
#-------------------------------------------------------------------------------------------
#---------------------------Visual Cues based----------------------------------------------------------------
    colorizer = Colorizer(
        gray_image_file='temp_image.jpg',
        visual_clues_file='cue_image.jpg'
    )
    result = colorizer.colorize()
    cv2.imwrite("visual_cue_result.jpg", result)
#------------------------------------------------------------------------------------------------
#-------------------------CNN based----------------------------------------------------------------------
# load colorizers
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
   
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img("temp_image.jpg")
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    # cv2.imwrite("out_img_eccv16.jpg", out_img_eccv16)
    # cv2.imwrite("out_img_siggraph17.jpg", out_img_siggraph17)
    plt.imsave('saved_eccv16.png', out_img_eccv16)
    plt.imsave('saved_siggraph17.png', out_img_siggraph17)
#-------------------------------------------------------------------------------------------
#-----------------------------Calculating MAE----------------------------------------------------
    maeResults = []
    namesOfAlgo = ["CV2Default","KNN_ExampleBased","Visual_cues_based","CNN_ECCV16","CNN_Siggraph17"]
    cv2defaultmae = measureMae("ref_image.jpg","colorized_image.jpg")
    knnmae = measureMae("ref_image.jpg","outputs/predicted_image.jpg")
    visualcuesmae = measureMae("ref_image.jpg","visual_cue_result.jpg")
    eccv16mae = measureMae("ref_image.jpg","saved_eccv16.png")
    siggmae = measureMae("ref_image.jpg","saved_siggraph17.png")

    maeResults.append(cv2defaultmae)
    maeResults.append(knnmae)
    maeResults.append(visualcuesmae)
    maeResults.append(eccv16mae)
    maeResults.append(siggmae)
#-----------------------------return everything---------------------------------------------------
    # Return the path of the colorized image
    return jsonify({
        'result': "../backend/"+colorized_filename,
        'resultKNN': "../backend/outputs/predicted_image.jpg",
        'resultCue': "../backend/visual_cue_result.jpg",
        'resultEccv': "../backend/saved_eccv16.png",
        'resultSigg': "../backend/saved_siggraph17.png",
        'mae' : maeResults,
        'names':namesOfAlgo
        })
# frontend/backend/colorized_image.jpg

if __name__ == '__main__':
    app.run(debug=True)
