import React, { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Box, Button, Typography, Paper } from '@mui/material';
import { styled } from '@mui/material/styles';
import { FiberManualRecord, Stop, NavigateNext } from '@mui/icons-material';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
  backgroundColor: '#f5f5f5',
  borderRadius: '15px',
}));

const RecordButton = styled(Button)(({ theme, isrecording }) => ({
  backgroundColor: isrecording === 'true' ? '#ff4444' : '#4CAF50',
  color: 'white',
  '&:hover': {
    backgroundColor: isrecording === 'true' ? '#ff0000' : '#45a049',
  },
  margin: theme.spacing(2),
}));

const VideoRecorder = ({ onRecordingComplete }) => {
  const webcamRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [currentSign, setCurrentSign] = useState(1);
  const [signFrames, setSignFrames] = useState([]);
  const [currentSignFrames, setCurrentSignFrames] = useState([]);
  const frameInterval = useRef(null);

  const captureFrame = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setCurrentSignFrames(prev => [...prev, imageSrc]);
      }
    }
  }, []);

  const handleStartRecording = useCallback(() => {
    setIsRecording(true);
    setSignFrames([]);
    setCurrentSignFrames([]);
    setCurrentSign(1);
    
    // Capture frames at regular intervals (30fps)
    frameInterval.current = setInterval(captureFrame, 33);
  }, [captureFrame]);

  const handleNextSign = useCallback(() => {
    // Save current sign frames
    setSignFrames(prev => [...prev, currentSignFrames]);
    // Reset for next sign
    setCurrentSignFrames([]);
    setCurrentSign(prev => prev + 1);
  }, [currentSignFrames]);

  const handleStopRecording = useCallback(() => {
    setIsRecording(false);
    clearInterval(frameInterval.current);
    
    // Save last sign's frames
    const allFrames = [...signFrames, currentSignFrames];
    
    // Process all captured frames
    if (allFrames.length > 0) {
      onRecordingComplete(allFrames);
    }
    
    // Reset states
    setSignFrames([]);
    setCurrentSignFrames([]);
    setCurrentSign(1);
  }, [signFrames, currentSignFrames, onRecordingComplete]);

  return (
    <StyledPaper elevation={3}>
      <Box sx={{ position: 'relative' }}>
        <Webcam
          ref={webcamRef}
          audio={false}
          screenshotFormat="image/jpeg"
          style={{
            width: '100%',
            borderRadius: '10px',
          }}
        />
        
        {isRecording && (
          <Typography
            variant="h5"
            sx={{
              position: 'absolute',
              bottom: 20,
              left: '50%',
              transform: 'translateX(-50%)',
              color: 'white',
              backgroundColor: 'rgba(0,0,0,0.7)',
              padding: '10px 20px',
              borderRadius: '20px',
            }}
          >
            Making Sign #{currentSign}
          </Typography>
        )}
      </Box>
      
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
        <RecordButton
          variant="contained"
          startIcon={isRecording ? <Stop /> : <FiberManualRecord />}
          onClick={isRecording ? handleStopRecording : handleStartRecording}
          isrecording={isRecording.toString()}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </RecordButton>
        
        {isRecording && (
          <Button
            variant="contained"
            color="primary"
            endIcon={<NavigateNext />}
            onClick={handleNextSign}
          >
            Next Sign
          </Button>
        )}
      </Box>
      
      {isRecording && (
        <Typography
          variant="body1"
          sx={{ textAlign: 'center', mt: 2, color: 'text.secondary' }}
        >
          Make your sign and click "Next Sign" to continue, or "Stop Recording" when finished
        </Typography>
      )}
    </StyledPaper>
  );
};

export default VideoRecorder;
