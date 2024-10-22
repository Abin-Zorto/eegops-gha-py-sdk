#!/bin/bash

# Navigate to the actions directory
cd .github/workflows/actions

# Loop through each file
for file in *.yml
do
    # Remove the .yml extension to get the action name
    action_name="${file%.yml}"
    
    # Create a new directory for the action
    mkdir -p "../../actions/$action_name"
    
    # Copy the content of the file to a new action.yml in the new directory
    cp "$file" "../../actions/$action_name/action.yml"
    
    # Delete the original file
    rm "$file"
    
    echo "Processed $file"
done

# Navigate back to the root of the repository
cd ../../..

echo "Migration completed!"