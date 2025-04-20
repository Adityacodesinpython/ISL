import React, { useState } from 'react';
import { Container, Typography, Box } from '@mui/material';
import VideoRecorder from './components/VideoRecorder';
import PredictionDisplay from './components/PredictionDisplay';
import axios from 'axios';

function App() {
  const [predictions, setPredictions] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);

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
      setPredictions(results);
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
    </Container>
  );
}

export default App;
