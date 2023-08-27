import React, { useState } from 'react';

import { styled } from "@mui/material/styles";
import Box from "@mui/material/Box";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import IconButton from "@mui/material/IconButton";
import SaveIcon from "@mui/icons-material/Save";
import DeleteIcon from "@mui/icons-material/Delete";
import TextField from "@mui/material/TextField";

import axios from 'axios';

const Demo = styled("div")(({ theme }) => ({
  backgroundColor: theme.palette.background.paper
}));

function CommentWall() {
  const [editIndex, setEditIndex] = React.useState(-1);
  const [editText, setEditText] = React.useState("");
  const [listItems, setListItems] = React.useState([]);
  const [newComment, setNewComment] = useState("");
    
  function fetchAPI(text) {
    axios.post('http://localhost:5000/intent', {
          input: text
    })
    .then(response => {
        setEditText(response.data.results);
        setListItems(prevListItems => [...prevListItems, response.data.results]);
    })
    .catch(error => console.error("Error:", error));
};


  const handleInputChange = (event) => {
    setNewComment(event.target.value);
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      if (newComment.trim() !== "") {
        fetchAPI(newComment.trim());
        setNewComment("");
      }
      event.preventDefault(); // Prevents the addition of a newline character when Enter is pressed
    } else if (event.key === "Tab") {
      event.preventDefault();
      setNewComment((prevComment) => prevComment + "\t");
    }
  };

  const handleTextKeyDown = (event) => {
    if (event.key === "Enter" && event.shiftKey) {
      event.preventDefault();
      setEditText((editText) => editText + "\n");
    } else if (event.key === "Tab") {
      event.preventDefault();
      setEditText((editText) => editText + "\t");
    }
  };

  const handleTextClick = (index, text) => {
    setEditIndex(index);
    setEditText(text);
  };

  const handleTextChange = (event) => {
    setEditText(event.target.value);
  };

  const handleTextSave = (index) => {
    const updatedListItems = [...listItems];
    updatedListItems[index] = editText;
    setListItems(updatedListItems);
    setEditIndex(-1);
  };

  const handleSaveClick = () => {
    const textToSave = listItems.join("\n");
    const blob = new Blob([textToSave], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = "VideoComments.txt";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDeleteClick = (index) => {
    const updatedListItems = listItems.filter((_, i) => i !== index);
    setListItems(updatedListItems);
  };

  return (
    <Demo>
      <Box sx={{ flexGrow: 1, maxWidth: "fullWidth" }}>
        <TextField
          fullWidth
          id="fullWidth"
          multiline
          label="Comment"
          value={newComment}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          InputProps={{
            endAdornment: (
              <IconButton edge="end" onClick={handleSaveClick}>
                <SaveIcon />
              </IconButton>
            )
          }}
        ></TextField>
        <List>
          {listItems.map((text, index) => (
            <ListItem key={index}>
              <div
                style={{
                  fontSize: "inherit",
                  width: "100%",
                  padding: "0 0",
                  fontFamily: "inherit",
                  letterSpacing: "inherit",
                  whiteSpace: "pre-line"
                }}
              >
                {editIndex === index ? (
                  <input
                    type="text"
                    value={editText}
                    onChange={handleTextChange}
                    onKeyDown={handleTextKeyDown}
                    onBlur={() => handleTextSave(index)}
                    autoFocus
                    style={{
                      fontSize: "inherit",
                      width: "100%",
                      padding: "0.5rem 0",
                      margin: 0,
                      background: "transparent",
                      caretColor: "black",
                      fontFamily: "inherit",
                      letterSpacing: "inherit"
                    }}
                  />
                ) : (
                  <div
                    onClick={() => handleTextClick(index, text)}
                    style={{ cursor: "text" }}
                  >
                    {text}
                  </div>
                )}
              </div>
              {listItems.length > 0 && (
                <IconButton
                  edge="end"
                  aria-label="delete"
                  onClick={() => handleDeleteClick(index)}
                >
                  <DeleteIcon />
                </IconButton>
              )}
            </ListItem>
          ))}
        </List>
      </Box>
    </Demo>
  );
}
export default CommentWall;