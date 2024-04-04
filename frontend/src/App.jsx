import React, { useState } from 'react';
import axios from 'axios';
import './App.css'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';


function App() {
  const [image, setImage] = useState(null);
  const [imageRef, setImageRef] = useState(null);
  const [imageCue, setImageCue] = useState(null);
  const [previewURL, setPreviewURL] = useState('');
  const [previewURLRef, setPreviewURLRef] = useState('');
  const [previewURLCue, setPreviewURLCue] = useState('');
  const [results, setResults] = useState("");
  const [resultsRef, setResultsRef] = useState("");
  const [resultsCue, setResultsCue] = useState("");
  const [resultsEccv, setResultsEccv] = useState("");
  const [resultsSigg, setResultsSigg] = useState("");
  const [maeData, setMaeData] = useState([]);

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
  const handleImageChangeCue = (event) => {
    const selectedImage = event.target.files[0];
    setImageCue(selectedImage);

    // Generate a preview URL for the selected image
    const imageURL = URL.createObjectURL(selectedImage);
    setPreviewURLCue(imageURL);
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('image', image);
    formData.append('ref', imageRef);
    formData.append('cue', imageCue);

    try {
      const response = await axios.post('http://localhost:5000/colorize', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setResults(response.data.result);
      setResultsRef(response.data.resultKNN);
      setResultsCue(response.data.resultCue);
      setResultsEccv(response.data.resultEccv);
      setResultsSigg(response.data.resultSigg);
      console.log(response);
      console.log(response.data.result);
      setMaeData(response.data.mae.map((value, index) => ({
        name: response.data.names[index],
        mae: value
      })));
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
      <p>Upload Visual Cue</p>
      <input type="file" accept="image/*" onChange={handleImageChangeCue} />
      {previewURLCue && <img src={previewURLCue} alt="PreviewCue" style={{ maxWidth: '100%', maxHeight: '200px' }} />}

      <button onClick={handleUpload}>Upload</button>
      <div>
        <p>colored by cv2 default</p>
        <img src={results}  />
      </div>
      <div>
        <p>colored by KNN example based</p>
        <img src={resultsRef}  />
      </div>
      <div>
        <p>colored by Visual Cues Algo</p>
        <img src={resultsCue}  />
      </div>
      <div>
        <p>colored by ECCV16</p>
        <img src={resultsEccv}  />
      </div>
      <div>
        <p>colored by Siggraph16</p>
        <img src={resultsSigg}  />
      </div>
      <BarChart width={800} height={400} data={maeData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="mae" fill="#8884d8" />
      </BarChart>
    </div>
  );
}

export default App;
