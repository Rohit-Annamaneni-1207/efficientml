import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def latency_metrics(total_time, num_samples):
    return {
        "total_inference_time": total_time,
        "avg_latency_ms": (total_time / num_samples) * 1000,
        "throughput_samples_per_sec": num_samples / total_time,
    }