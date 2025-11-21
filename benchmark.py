#!/usr/bin/env python3
"""
Benchmark script for Modern Assyrian GPT API.
Measures latency, tokens/second, and response consistency.
"""
import requests
import time
import statistics
import json
from typing import List, Dict

API_URL = "http://localhost:8000"

def test_endpoint_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def benchmark_generate(
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.8,
    num_runs: int = 10
) -> Dict:
    """
    Run multiple generation requests and collect metrics.
    
    Returns:
        Dict with latency stats, tokens/sec, and sample outputs.
    """
    latencies = []
    tokens_generated = []
    outputs = []
    
    print(f"Running {num_runs} requests with prompt: '{prompt[:50]}...'")
    
    for i in range(num_runs):
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        start = time.time()
        try:
            response = requests.post(
                f"{API_URL}/generate",
                json=payload,
                timeout=30
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                latencies.append(elapsed)
                
                # Extract token count from generated text (approximation)
                # Better: parse from API response if available
                gen_text = data.get("generated_text", "")
                token_count = len(gen_text.split())  # Word-based approximation
                tokens_generated.append(token_count)
                
                if i == 0:  # Store first output as sample
                    outputs.append(gen_text[:200])
                
                print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s, ~{token_count} tokens")
            else:
                print(f"  Run {i+1}/{num_runs}: ERROR {response.status_code}")
        except requests.RequestException as e:
            print(f"  Run {i+1}/{num_runs}: FAILED - {e}")
    
    if not latencies:
        return {"error": "All requests failed"}
    
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    avg_tokens = statistics.mean(tokens_generated) if tokens_generated else 0
    tokens_per_sec = avg_tokens / avg_latency if avg_latency > 0 else 0
    
    return {
        "num_successful_runs": len(latencies),
        "latency": {
            "mean": avg_latency,
            "median": median_latency,
            "std_dev": std_latency,
            "min": min_latency,
            "max": max_latency,
            "unit": "seconds"
        },
        "tokens": {
            "avg_generated": avg_tokens,
            "tokens_per_second": tokens_per_sec
        },
        "sample_output": outputs[0] if outputs else None
    }

def check_metrics_endpoint() -> Dict:
    """Fetch current metrics from API."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Status {response.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}

def main():
    print("=" * 60)
    print("Modern Assyrian GPT API Benchmark")
    print("=" * 60)
    print()
    
    # Health check
    print("1. Checking API health...")
    if not test_endpoint_health():
        print("   ❌ API not available. Start the server first:")
        print("      docker-compose up -d")
        return
    print("   ✓ API is healthy\n")
    
    # Fetch initial metrics
    print("2. Initial metrics:")
    initial_metrics = check_metrics_endpoint()
    if "error" not in initial_metrics:
        print(f"   Total requests: {initial_metrics.get('total_requests', 0)}")
        print(f"   Total tokens: {initial_metrics.get('total_generated_tokens', 0)}")
        print(f"   Avg tokens/sec: {initial_metrics.get('avg_tokens_per_second', 0):.2f}")
    print()
    
    # Benchmark short prompt
    print("3. Benchmark: Short prompt (50 tokens)")
    short_results = benchmark_generate(
        prompt="ܫܠܡܐ",  # Peace/Hello in Assyrian
        max_tokens=50,
        temperature=0.8,
        num_runs=10
    )
    print()
    
    # Benchmark longer prompt
    print("4. Benchmark: Longer generation (100 tokens)")
    long_results = benchmark_generate(
        prompt="ܘܥܒ݂ܸܕܠܹܗ ܡܲܠܟܘܼܬܵܐ",
        max_tokens=100,
        temperature=0.7,
        num_runs=100
    )
    print()
    
    # Final metrics
    print("5. Final metrics:")
    final_metrics = check_metrics_endpoint()
    if "error" not in final_metrics:
        print(f"   Total requests: {final_metrics.get('total_requests', 0)}")
        print(f"   Total tokens: {final_metrics.get('total_generated_tokens', 0)}")
        print(f"   Avg tokens/sec: {final_metrics.get('avg_tokens_per_second', 0):.2f}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if "error" not in short_results:
        print(f"\nShort prompt (50 tokens):")
        print(f"  Avg latency: {short_results['latency']['mean']:.3f}s ± {short_results['latency']['std_dev']:.3f}s")
        print(f"  Tokens/sec:  {short_results['tokens']['tokens_per_second']:.2f}")
        print(f"  Sample:      {short_results['sample_output'][:80]}...")
    
    if "error" not in long_results:
        print(f"\nLonger generation (100 tokens):")
        print(f"  Avg latency: {long_results['latency']['mean']:.3f}s ± {long_results['latency']['std_dev']:.3f}s")
        print(f"  Tokens/sec:  {long_results['tokens']['tokens_per_second']:.2f}")
        print(f"  Sample:      {long_results['sample_output'][:80]}...")
    
    print()
    print("Run complete. Use these results to track performance over time.")
    print()

if __name__ == "__main__":
    main()
