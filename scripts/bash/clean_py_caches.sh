#!/bin/bash

# Function to remove __pycache__ directories
remove_pycache() {
  find . -type d -name "__pycache__" -exec rm -r {} +
}

# Call the function
remove_pycache


