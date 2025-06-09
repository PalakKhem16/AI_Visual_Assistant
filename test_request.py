import requests
import os

# Set Tesseract path in environment
os.environ['PATH'] = os.environ['PATH'] + ';C:\\Program Files\\Tesseract-OCR'

def test_image_processing(image_path: str):
    """
    Test the image processing endpoint with any image file.
    
    Args:
        image_path: Path to the image file to test (e.g., 'pic.png', 'pic2.png')
    """
    url = 'http://localhost:8000/process-frame/'
    
    try:
        print(f"\nTesting with image: {image_path}")
        # Open and send the image with a 10 second timeout
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f, 'image/png')}
            response = requests.post(url, files=files, timeout=10)
        
        # Handle the response
        if response.status_code == 200:
            content = response.json()
            print("\n✅ Success!")
            print("\nDetected Objects:")
            for obj in content.get('detected_objects', []):
                print(f"- {obj['class']} (confidence: {obj['confidence']:.2f})")
            print(f"\nDescription: {content.get('description', 'No description')}")
            print(f"Audio file: {content.get('audio_url', 'No audio')}")
        else:
            print(f"❌ Error: Status code {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    # You can test any image by providing its path
    test_image_processing('pic.png')  # Test with pic.png
    test_image_processing('pic2.png')  # Test with pic2.png
    test_image_processing('pic3.png') 
