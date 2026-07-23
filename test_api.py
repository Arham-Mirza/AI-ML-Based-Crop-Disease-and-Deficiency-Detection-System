import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    try:
        # Test home endpoint
        print("1. Testing home endpoint...")
        response = requests.get(f"{base_url}/")
        print("Home:", json.dumps(response.json(), indent=2))
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health")
        print("Health:", json.dumps(response.json(), indent=2))
        
        # Test classes endpoint
        print("\n3. Testing classes endpoint...")
        response = requests.get(f"{base_url}/classes")
        print("Classes:", json.dumps(response.json(), indent=2))
        
        # Test prediction with file upload
        print("\n4. Testing prediction with file upload...")
        try:
            with open('example.jpg', 'rb') as f:
                files = {'file': ('example.jpg', f, 'image/jpeg')}
                response = requests.post(f"{base_url}/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("Prediction Results:")
                print(f"  🍎 Fruit: {result.get('fruit_type', 'Unknown')}")
                print(f"  ✅ Health: {'Healthy' if result.get('is_healthy') else 'Rotten'}")
                print(f"  🎯 Confidence: {result.get('confidence', 0):.2%}")
                print(f"  📊 Full Prediction: {result.get('prediction', 'Unknown')}")
                
                print("\n  Top 3 Predictions:")
                for pred in result.get('top_predictions', []):
                    print(f"    - {pred['class']}: {pred['confidence']:.2%}")
            else:
                print(f"Prediction failed: {response.json()}")
                
        except FileNotFoundError:
            print("example.jpg not found - skipping file upload test")
        
        # Test local prediction endpoint
        print("\n5. Testing local prediction endpoint...")
        response = requests.get(f"{base_url}/predict_local")
        if response.status_code == 200:
            result = response.json()
            print("Local Prediction Results:")
            print(f"  🍎 Fruit: {result.get('fruit_type', 'Unknown')}")
            print(f"  ✅ Health: {'Healthy' if result.get('is_healthy') else 'Rotten'}")
            print(f"  🎯 Confidence: {result.get('confidence', 0):.2%}")
        else:
            print(f"Local prediction failed: {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {base_url}")
        print("Make sure the server is running with: python app.py")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

if __name__ == "__main__":
    test_api()