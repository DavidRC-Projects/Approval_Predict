import os
import sys

def setup_project_path():
    """Set up the project path for imports and data access"""
    current_dir = os.getcwd()
    
    if 'jupyter_notebooks' in current_dir:
        os.chdir(os.path.dirname(current_dir))
        print("Changed to project root directory")
    else:
        print("No 'jupyter_notebooks' directory found the the current path")
    
    # Add project root to Python path for imports
    project_root = os.getcwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return os.getcwd()