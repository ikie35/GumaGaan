from roboflow import Roboflow

def main():
    rf = Roboflow(api_key="tsstVHnAqPFAhbWOYzrg")
    workspace = rf.workspace("rubber-research")
    project = workspace.project("yanglabel7class-v2-budv6")
    
    for ver_num in [1, 2]:
        print(f"\nFetching info for Version {ver_num}...")
        version = project.version(ver_num)
        # Class list is usually in version.yaml or dataset details
        # Let's inspect the version object fields
        print(f"  Version number: {version.version}")
        print(f"  Created: {version.created}")
        print(f"  Images: {version.images}")
        # Note: version.download() is standard, let's see if we can access model or class list
        # We can also check details from the raw JSON response
        # Roboflow version class doesn't always expose .classes, let's check dict representation
        try:
            print(f"  Check fields: {list(version.__dict__.keys())}")
            # If there's a structure or raw json, let's print it
            if hasattr(version, 'name'):
                print(f"  Name: {version.name}")
        except Exception as e:
            print(f"  Could not list fields: {e}")

if __name__ == "__main__":
    main()
