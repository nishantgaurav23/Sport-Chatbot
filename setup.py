import os
import sys

def create_project_structure():
    """Create the project directory structure and files"""
    
    # Create directories
    directories = [
        'ESPN_data',
        'embeddings_cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("HUGGINGFACE_API_KEY=your_api_key_here\n")
        print("Created .env file")

    # Create .gitignore if it doesn't exist
    if not os.path.exists('.gitignore'):
        gitignore_content = """
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Virtual Environment
venv/
env/
ENV/

# Cache directories
embeddings_cache/
.cache/

# IDE specific files
.vscode/
.idea/

# Operating System
.DS_Store
Thumbs.db
"""
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content.strip())
        print("Created .gitignore file")

    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Add your HuggingFace API key to the .env file")
    print("2. Place your ESPN data CSV files in the ESPN_data directory")
    print("3. Install requirements: pip install -r requirements.txt")
    print("4. Run embedding generation: python embedding_processor.py")
    print("5. Start the app: streamlit run app.py")

if __name__ == "__main__":
    create_project_structure()