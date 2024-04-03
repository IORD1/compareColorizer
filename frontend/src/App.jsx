import React, { useState } from 'react';
import axios from 'axios';
import './App.css'

function App() {
  const [image, setImage] = useState(null);
  const [imageRef, setImageRef] = useState(null);
  const [previewURL, setPreviewURL] = useState('');
  const [previewURLRef, setPreviewURLRef] = useState('');
  const [results, setResults] = useState("");
  const [resultsRef, setResultsRef] = useState("");

  const handleImageChange = (event) => {
    const selectedImage = event.target.files[0];
    setImage(selectedImage);

    // Generate a preview URL for the selected image
    const imageURL = URL.createObjectURL(selectedImage);
    setPreviewURL(imageURL);
  };
  const handleImageChangeRef = (event) => {
    const selectedImage = event.target.files[0];
    setImageRef(selectedImage);

    // Generate a preview URL for the selected image
    const imageURL = URL.createObjectURL(selectedImage);
    setPreviewURLRef(imageURL);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('ref', imageRef);

    try {
      const response = await axios.post('http://localhost:5000/colorize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResults(response.data.result);
      setResultsRef(response.data.resultKNN);
      console.log(response);
      console.log(response.data.result);

    } catch (error) {
      console.error('Error uploading image:', error);
    }
  };

  return (
    <div>
      <p>Upload B/W</p>
      <input type="file" accept="image/*" onChange={handleImageChange} />
      {previewURL && <img src={previewURL} alt="Preview" style={{ maxWidth: '100%', maxHeight: '200px' }} />}
      <p>Upload reference colored</p>
      <input type="file" accept="image/*" onChange={handleImageChangeRef} />
      {previewURLRef && <img src={previewURLRef} alt="PreviewRef" style={{ maxWidth: '100%', maxHeight: '200px' }} />}
      <button onClick={handleUpload}>Upload</button>
      <div>
        <p>colored by cv2 default</p>
        <img src={results}  />
      </div>
      <div>
        <p>colored by KNN example based</p>
        <img src={resultsRef}  />
      </div>
    </div>
  );
}

export default App;
