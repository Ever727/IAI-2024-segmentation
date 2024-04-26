import os
import multiprocessing
import concurrent.futures

python_path = (
    "C:\\Users\\qq642\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
)
learning_rates = [1e-4, 7e-5, 5e-5, 1e-5]
batch_sizes = [32, 64, 128]
epochs = [15]
max_lens = [32, 64, 128]
models = ["RNN_LSTM", "RNN_GRU", "TextCNN", "MLP", "Transformer", "Bert"]

MAX_PROCESSES = 5  # 设置最大并行进程数量
MAX_THREADS = 10  # 设置最大并发线程数量


def run_experiment(learning_rate, batch_size, epoch, max_len, model):
    print(
        f"Experiment: {model}, LR: {learning_rate}, BS: {batch_size}, Epoch: {epoch}, Max Length: {max_len}"
    )
    os.system(
        f"{python_path} ./src/main.py -n {model} -l {learning_rate} -b {batch_size} -e {epoch} -m {max_len}"
    )


def pipeline():
    processes = []
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for max_len in max_lens:
                    for model in models:
                        process = multiprocessing.Process(
                            target=run_experiment,
                            args=(learning_rate, batch_size, epoch, max_len, model),
                        )
                        process.start()
                        processes.append(process)

                        # 控制并行进程数量，当达到最大进程数量时，等待进程完成后再启动新的进程
                        if len(processes) >= MAX_PROCESSES:
                            for p in processes:
                                p.join()
                            processes = []

    for process in processes:
        process.join()


def pipeline2():
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = []
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    for max_len in max_lens:
                        for model in models:
                            future = executor.submit(
                                run_experiment,
                                learning_rate,
                                batch_size,
                                epoch,
                                max_len,
                                model,
                            )
                            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)


def run_single_experiment(model, learning_rate, batch_size, epoch, max_len):
    print(
        f"Experiment: {model}, LR: {learning_rate}, BS: {batch_size}, Epoch: {epoch}, Max Length: {max_len}"
    )
    os.system(
        f"{python_path} ./src/main.py -n {model} -l {learning_rate} -b {batch_size} -e {epoch} -m {max_len}"
    )


if __name__ == "__main__":
    # pipeline2()

    for leraning_rate in learning_rates:
        for batch_size in batch_sizes:
            for epoch in epochs:
                for max_len in max_lens:
                    run_single_experiment(
                        "Bert", leraning_rate, batch_size, epoch, max_len
                    )

    # run_single_experiment("Bert", 5e-5, 64, 15, 64)
