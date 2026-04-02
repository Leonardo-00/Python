import os

class Generator:
    
    def __init__(self):
        
        file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

        # Check if the file exists
        if os.path.exists(file_path):
            self.file = open(file_path, "a")  # Open in append mode if it exists
        else:
            self.file = open(file_path, "x")  # Create a new file if it doesn't exist
    
    

