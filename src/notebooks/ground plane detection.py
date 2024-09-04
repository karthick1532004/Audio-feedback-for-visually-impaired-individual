import sys
from PIL import Image

def extract_depth_info_from_jpeg(file_path):
    try:
        img = Image.open(file_path)
        depth_info = img.info.get('depth')  # Extract the 'depth' metadata, if available

        if depth_info is not None:
            return depth_info
        else:
            return None
    except Exception as e:
        print("Error: ", str(e))
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python depth_extraction.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    depth_info = extract_depth_info_from_jpeg(image_path)

    if depth_info is not None:
        print("Depth information found:")
        print(depth_info)
    else:
        print("No depth information found in the JPEG.")
