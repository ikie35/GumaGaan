import json
from pathlib import Path

def main():
    notebook_path = Path(r"D:\school\Masters\Research\Training\Two stage Rubber Disease\Stage2_Classifier\kaggle_stage2_notebook.ipynb")
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    updated = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for i, line in enumerate(source):
                if 'stage-rubber-diseases-classifier' in line:
                    source[i] = line.replace('stage-rubber-diseases-classifier', 'stage-2-rubber-leaf-classifier')
                    updated = True
                if 'project.version(1)' in line:
                    source[i] = line.replace('project.version(1)', 'project.version(2)')
                    updated = True
                    
    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
        print("Notebook updated successfully!")
    else:
        print("No changes needed in notebook.")

if __name__ == '__main__':
    main()
