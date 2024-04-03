from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import os
import skimage
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



@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    image_file_ref = request.files['ref']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})


    # Save the uploaded image temporarily
    temp_filename = 'temp_image.jpg'
    ref_image = 'ref_image.jpg'
    image_file.save(temp_filename)
    image_file_ref.save(ref_image)
    #-----------------------------------------------------------------------
            #temp_filename = 'temp_image.jpg'
            #ref_image = 'ref_image.jpg'

            
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
#---------------------------CNN based----------------------------------------------------------------


#-----------------------------return everything---------------------------------------------------
    # Return the path of the colorized image
    return jsonify({
        'result': "../backend/"+colorized_filename,
        'resultKNN': "../backend/outputs/predicted_image.jpg",
        })
# frontend/backend/colorized_image.jpg

if __name__ == '__main__':
    app.run(debug=True)
