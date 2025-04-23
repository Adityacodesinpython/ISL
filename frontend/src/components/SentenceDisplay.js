import React from 'react';
import { List, ListItem, ListItemText, Typography } from '@mui/material';

function SentenceDisplay({ sentences }) {
  return (
    <List>
      {sentences.map((sentence, index) => (
        <ListItem key={index}>
          <ListItemText
            primary={
              <Typography variant="h6">
                Sentence: {sentence}
              </Typography>
            }
          />
        </ListItem>
      ))}
    </List>
  );
}

export default SentenceDisplay;
