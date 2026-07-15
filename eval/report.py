def print_report(results):

    print("\n===================================")
    print("Benchmark Report")
    print("===================================")

    print(f"Accuracy          : {results['accuracy']:.4f}")
    print(f"Model Size        : {results['model_size_mb']:.2f} MB")
    print(f"Total Time        : {results['total_time']:.3f} s")
    print(f"Avg Latency       : {results['avg_latency'] * 1000:.3f} ms/sample")
    print(f"Throughput        : {results['throughput']:.2f} samples/s")

    if "compression_ratio" in results:
        print(f"Compression Ratio : {results['compression_ratio']:.2f}x")

    print("===================================\n")