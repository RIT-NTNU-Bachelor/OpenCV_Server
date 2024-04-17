import os
import sys

# Adds the parent 'src' directory to the system path.
# This allows the test files to import modules from the src directory.
def set_project_path_for_tests():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '../src')  
    sys.path.insert(0, parent_dir)
