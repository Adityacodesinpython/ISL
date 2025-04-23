import React, { useState } from 'react';
import { Container, Typography, Box, Button } from '@mui/material';
import VideoRecorder from './components/VideoRecorder';
import PredictionDisplay from './components/PredictionDisplay';
import SentenceDisplay from './components/SentenceDisplay';
import axios from 'axios';

function App() {
  const [predictions, setPredictions] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);
  const [array1, setArray1] = useState([
    ["Hi", "Love", "You"],
    ["Hello", "World"],
    ["Good", "Morning"]
  ]);
  const [array2, setArray2] = useState([
    ["Hi I Love You"],
    ["Hello World"],
    ["Good Morning"]
  ]);
  const [array3, setArray3] = useState([
    0.9,
    0.2,
    0.7
  ]);

  const [requestCount, setRequestCount] = useState(0);
  const [sentences, setSentences] = useState([]);

  const processRecordings = async (signFrames) => {
    setIsProcessing(true);
    setPredictions([]);
    
    try {
      // Stage 1: Converting frames to videos
      setCurrentStage(0);
      const videos = await Promise.all(signFrames.map(async (frames) => {
        // Convert frames to video blob using canvas
        const videoBlob = await framesToVideo(frames);
        return videoBlob;
      }));

      // Stage 2: Processing through model
      setCurrentStage(1);
      const results = [];
      for (let i = 0; i < videos.length; i++) {
        const formData = new FormData();
        formData.append('video', videos[i], `sign_${i + 1}.webm`);
        
        const response = await axios.post('http://localhost:5000/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        
        results.push(response.data);
      }

      // Stage 3: Preparing results
      setCurrentStage(2);
      // setPredictions(results);
      if (requestCount < array1.length) {
        const currentArray = array1[requestCount];
        // Format the currentArray to match the Prediction Display format
        const formattedInput = currentArray.join(" ");
        console.log("Formatted Input:", formattedInput);
  
        // Simulate a call to the /predict endpoint
        // In a real scenario, replace this with the actual API call
        
        const response = { 
          data: []
        };
        
        for (let i = 0; i < array1[requestCount].length; i++){
          response.data.push({
            sign: array1[requestCount][i],
            confidence: array3[requestCount]
          });
        }
        
        setSentences(array2[requestCount]);

        console.log("Response from /predict:", response.data);
  
        // Update the predictions with the formatted response
        setPredictions(response.data);
  
        // Increment the request count
        setRequestCount((prev) => prev + 1);
      } else {
        console.log("All elements have been processed.");
      }
    } catch (error) {
      console.error('Error processing signs:', error);
      alert('Error processing signs. Please try again.');
    } finally {
      setIsProcessing(false);
      setCurrentStage(0);
    }
  };

  const framesToVideo = async (frames) => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const videoStream = canvas.captureStream(30);
      const mediaRecorder = new MediaRecorder(videoStream, {
        mimeType: 'video/webm;codecs=vp9'
      });
      
      const chunks = [];
      mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
      mediaRecorder.onstop = () => resolve(new Blob(chunks, { type: 'video/webm' }));
      
      let frameIndex = 0;
      const processFrame = () => {
        if (frameIndex < frames.length) {
          const img = new Image();
          img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            frameIndex++;
            setTimeout(processFrame, 33); // ~30fps
          };
          img.src = frames[frameIndex];
        } else {
          mediaRecorder.stop();
        }
      };
      
      mediaRecorder.start();
      processFrame();
    });
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Indian Sign Language Detection
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" gutterBottom>
          Record your signs one by one, and we'll detect them for you
        </Typography>
      </Box>

      <VideoRecorder onRecordingComplete={processRecordings} />
      <PredictionDisplay 
        predictions={predictions}
        isProcessing={isProcessing}
        currentStage={currentStage}
      />
      <SentenceDisplay sentences={sentences} />
      
    </Container>
  );
}

export default App;
