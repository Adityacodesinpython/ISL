import React from 'react';
import { Box, Paper, Typography, List, ListItem, ListItemText, CircularProgress } from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginTop: theme.spacing(3),
  backgroundColor: '#f5f5f5',
  borderRadius: '15px',
}));

const ProcessingStage = styled(Box)(({ theme, active }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  backgroundColor: active ? theme.palette.primary.light : theme.palette.grey[300],
  color: active ? theme.palette.primary.contrastText : theme.palette.text.secondary,
  borderRadius: theme.spacing(1),
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(2),
}));

const PredictionDisplay = ({ predictions, isProcessing, currentStage }) => {
  const stages = [
    'Converting recordings to videos',
    'Processing signs through model',
    'Preparing results'
  ];

  return (
    <>
      {(isProcessing || predictions.length > 0) && (
        <StyledPaper>
          {isProcessing && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Processing Status:
              </Typography>
              {stages.map((stage, index) => (
                <ProcessingStage 
                  key={index}
                  active={currentStage === index}
                >
                  {currentStage === index && <CircularProgress size={20} color="inherit" />}
                  <Typography>
                    {stage}
                  </Typography>
                </ProcessingStage>
              ))}
            </Box>
          )}

          {predictions.length > 0 && (
            <>
              <Typography variant="h5" gutterBottom sx={{ color: 'primary.main' }}>
                Detected Signs:
              </Typography>
              <List>
                {predictions.map((pred, index) => (
                  <ListItem 
                    key={index}
                    sx={{
                      backgroundColor: 'white',
                      borderRadius: 1,
                      mb: 1,
                    }}
                  >
                    <ListItemText
                      primary={
                        <Typography variant="h6">
                          Sign #{index + 1}: {pred.sign}
                        </Typography>
                      }
                      secondary={
                        <Typography 
                          variant="body2"
                          color={pred.confidence > 0.7 ? 'success.main' : 'warning.main'}
                        >
                          Confidence: {(pred.confidence * 100).toFixed(2)}%
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </>
          )}
        </StyledPaper>
      )}
    </>
  );
};

export default PredictionDisplay;
