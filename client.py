import requests
import base64
import json

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image

def test_simple_endpoint():
    """Test the simple /recog endpoint (compatible with original client)"""
    url = 'http://127.0.0.1:8080/recog'
    
    # Test with vantoan image
    image_path = 'Dataset/raw/vantoan/OIP (1).jpg'
    encoded_image_data = encode_image_to_base64(image_path)
    
    data = {
        'image': encoded_image_data,
        'w': 100,  # width
        'h': 100   # height
    }
    
    response = requests.post(url, data=data)
    
    print("=== Simple Endpoint Test ===")
    print(f"Image: {image_path}")
    print(f"Response from server: {response.text}")
    print("-" * 40)

def test_detailed_endpoint():
    """Test the detailed /recog_detailed endpoint with JSON response"""
    url = 'http://127.0.0.1:8080/recog_detailed'
    
    # Test with vanhau image
    image_path = 'Dataset/raw/vanhau/OIP (1).jpg'
    encoded_image_data = encode_image_to_base64(image_path)
    
    data = {
        'image': encoded_image_data,
        'w': 100,  # width
        'h': 100   # height
    }
    
    response = requests.post(url, data=data)
    
    print("=== Detailed Endpoint Test ===")
    print(f"Image: {image_path}")
    print(f"Status Code: {response.status_code}")
    if response.headers.get('content-type') == 'application/json':
        response_json = response.json()
        print(f"Response JSON:")
        print(json.dumps(response_json, indent=2))
    else:
        print(f"Response: {response.text}")
    print("-" * 40)

def test_health_endpoint():
    """Test the health check endpoint"""
    url = 'http://127.0.0.1:8080/health'
    
    try:
        response = requests.get(url)
        print("=== Health Check Test ===")
        print(f"Status Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        print("-" * 40)
    except Exception as e:
        print(f"Health check failed: {e}")

def test_multiple_images():
    """Test with multiple images from both classes"""
    import os
    
    print("=== Multiple Images Test ===")
    
    test_images = []
    
    for class_name in ['vantoan', 'vanhau']:
        class_dir = f'Dataset/raw/{class_name}'
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in images[:2]:  # Test first 2 images from each class
                test_images.append((os.path.join(class_dir, img), class_name))
    
    url = 'http://127.0.0.1:8080/recog_detailed'
    
    for img_path, true_class in test_images:
        try:
            encoded_image_data = encode_image_to_base64(img_path)
            data = {
                'image': encoded_image_data,
                'w': 100,
                'h': 100
            }
            
            response = requests.post(url, data=data)
            if response.status_code == 200:
                result = response.json()
                predicted = result['name']
                confidence = result['confidence']
                status = "✅" if predicted == true_class else "❌"
                print(f"{status} {true_class} -> {predicted} (conf: {confidence:.3f})")
            else:
                print(f"❌ Error processing {img_path}: {response.text}")
                
        except Exception as e:
            print(f"❌ Error with {img_path}: {e}")

if __name__ == "__main__":
    print("Testing FaceNet Flask API")
    print("=" * 50)
    
    try:
        test_health_endpoint()
        test_simple_endpoint()
        test_detailed_endpoint()
        test_multiple_images()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Flask server.")
        print("Make sure the server is running on http://localhost:8080")
    except Exception as e:
        print(f"Error: {e}")
