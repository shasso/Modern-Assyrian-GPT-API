#!/usr/bin/env python3
"""
Test script for the Syriac GPT API
This mimics the testing done in section 4.3 of the notebook
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        return False

def test_generation(prompt, max_new_tokens=20, temperature=0.9, top_k=50):
    """Test text generation with given parameters"""
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k
    }

    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Prompt: {prompt}")
            print(f"Generated: {result['generated_text']}")
            print("-" * 50)
            return result['generated_text']
        else:
            print(f"❌ Generation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def run_tests():
    """Run the same tests as in section 4.3 of the notebook"""
    print("🧪 Testing Syriac GPT API")
    print("=" * 50)

    # Test health
    if not test_api_health():
        return

    # Test different prompts (similar to section 4.3)
    test_cases = [
        "ܘܓܵܪܕܵܐ ",  # Similar to the notebook
        "ܡܲܚܕܸܐ",   # Another test from the notebook
        "ܘܐ݇ܡܸܪܹܗ ܐܲܠܵܗܵܐ ",  # God said...
        "ܐܲܒ݂ܪܵܗܵܡ ܡܘܼܠܸܕ ܠܹܗ",  # Abraham begat...
    ]

    print("\n🧪 Running generation tests...")
    for prompt in test_cases:
        print(f"\nTesting prompt: '{prompt}'")
        # Run multiple times like in the notebook
        for i in range(3):
            result = test_generation(prompt, max_new_tokens=20, temperature=0.9, top_k=50)
            if result:
                time.sleep(0.5)  # Small delay between requests

    print("\n✅ All tests completed!")

if __name__ == "__main__":
    run_tests()