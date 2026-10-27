from roboflow import Roboflow
import urllib.request
import json

def main():
    api_key = "tsstVHnAqPFAhbWOYzrg"
    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace("rubber-research")
    project = workspace.project("yanglabel7class-v2-budv6")
    
    print(f"Project Name: {project.name}")
    
    # Fetch project classes
    url = f"https://api.roboflow.com/rubber-research/yanglabel7class-v2-budv6?api_key={api_key}"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            project_info = data.get('project', {})
            print(f"Project classes listed on Roboflow: {project_info.get('classes')}")
    except Exception as e:
        print(f"Error fetching project classes: {e}")
        
    for v_num in [1, 2, 3]:
        url = f"https://api.roboflow.com/rubber-research/yanglabel7class-v2-budv6/{v_num}?api_key={api_key}"
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                version_data = data.get('version', {})
                print(f"\nVersion {v_num}:")
                print(f"  Images: {version_data.get('images')}")
                print(f"  Preprocessing: {version_data.get('preprocessing')}")
                print(f"  Augmentation: {version_data.get('augmentation')}")
        except Exception as e:
            print(f"Error fetching version {v_num} info: {e}")

if __name__ == "__main__":
    main()
