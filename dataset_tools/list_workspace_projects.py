from roboflow import Roboflow

def main():
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    workspace = rf.workspace("rubber-research")
    print(f"Workspace: {workspace.name}")
    
    # List projects
    projects = workspace.projects()
    print("\nProjects in this workspace:")
    for p in projects:
        # p is like 'rubber-research/yanglabel7class-v2-budv6'
        proj_name = p.split('/')[-1]
        print(f"  - Project: {proj_name}")
        try:
            proj_obj = workspace.project(proj_name)
            print(f"    Type: {proj_obj.type} | Id: {proj_obj.id}")
            print(f"    Versions: {[v.version for v in proj_obj.versions()]}")
        except Exception as e:
            print(f"    Could not load project object: {e}")

if __name__ == "__main__":
    main()
