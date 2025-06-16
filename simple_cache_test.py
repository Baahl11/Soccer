from automatic_match_discovery import AutomaticMatchDiscovery
import time

print("=== CACHE PERFORMANCE TEST ===")
discovery = AutomaticMatchDiscovery()

print("Starting first run...")
start = time.time()
result = discovery.get_todays_predictions()
first_time = time.time() - start
print(f"First run: {first_time:.2f}s, {result.get('total_matches', 0)} matches")

print("Starting cached run...")
start = time.time()
result2 = discovery.get_todays_predictions()
second_time = time.time() - start
print(f"Cached run: {second_time:.2f}s, {result2.get('total_matches', 0)} matches")

if first_time > 0 and second_time > 0:
    speedup = first_time / second_time
    print(f"Speedup: {speedup:.1f}x faster with cache")
    print(f"Time saved: {first_time - second_time:.2f} seconds")
