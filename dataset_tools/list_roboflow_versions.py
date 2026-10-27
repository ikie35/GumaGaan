from roboflow import Roboflow

def main():
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    
    print("Loading workspace 'rubber-research'...")
    workspace = rf.workspace("rubber-research")
    
    print("Loading project 'yanglabel7class-v2-budv6'...")
    project = workspace.project("yanglabel7class-v2-budv6")
    
    print("\nProject Details:")
    print(f"Name: {project.name}")
    print(f"Type: {project.type}")
    print(f"Id: {project.id}")
    print(f"Versions list:")
    
    versions = project.versions()
    for v in versions:
        # v is a Version object
        print(f"  Version: {v.version} | Date: {v.created} | Images: {v.images}")

if __name__ == "__main__":
    main()
