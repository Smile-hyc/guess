#include "PCFG.h"
#include "md5.h"

#include <mpi.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace chrono;

struct GeneralConfig
{
    string train_path = "/guessdata/Rockyou-singleLined-full.txt";
    string test_path = "/guessdata/Rockyou-singleLined-full.txt";
    int test_limit = 1000000;

    long long max_guesses = 10000000;
    int hash_threshold = 1000000;
    int batch_size = 0; // 0 表示自动使用 MPI 进程数

    string local_mode = "serial"; // serial / openmp / pthread
    int local_threads = 1;

    string hash_mode = "scalar"; // scalar / simd
    bool enable_pt_level_mpi = true;
};

struct HashStats
{
    long long cracked = 0;
    long long hashed = 0;
    double time_hash = 0.0;
};

mutex hash_mutex;
condition_variable hash_cv;
deque<vector<string>> pending_hash_batches;
atomic<bool> hash_thread_should_exit(false);

static bool is_true_flag(const string &s)
{
    return s == "1" || s == "true" || s == "TRUE" || s == "yes" || s == "on";
}

static void print_usage(int rank)
{
    if (rank != 0)
    {
        return;
    }

    cout << "Usage: mpirun -np <N> ./correctness_guess_general [options]\n"
         << "Options:\n"
         << "  --train <path>             训练集路径，默认 /guessdata/Rockyou-singleLined-full.txt\n"
         << "  --test <path>              测试集路径，默认同训练集\n"
         << "  --test-limit <num>         测试集读取条数，默认 1000000\n"
         << "  --max-guesses <num>        全局候选口令生成上限，默认 10000000\n"
         << "  --hash-threshold <num>     生成多少候选后转交哈希线程，默认 1000000\n"
         << "  --batch-size <num>         每轮取出的 PT 数，默认等于 MPI 进程数\n"
         << "  --local-mode <mode>        进程内部生成方式：serial/openmp/pthread\n"
         << "  --local-threads <num>      进程内部线程数，默认 1\n"
         << "  --hash-mode <mode>         哈希方式：scalar/simd，默认 scalar\n"
         << "  --pt-mpi <0|1>             是否启用 PT 级 MPI 分配，默认 1\n"
         << "  --help                     打印帮助信息\n";
}

static GeneralConfig parse_args(int argc, char **argv, int rank)
{
    GeneralConfig cfg;

    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];
        auto need_value = [&](const string &name) -> string {
            if (i + 1 >= argc)
            {
                if (rank == 0)
                {
                    cerr << "Missing value for " << name << endl;
                }
                return "";
            }
            i++;
            return argv[i];
        };

        if (arg == "--help" || arg == "-h")
        {
            print_usage(rank);
            MPI_Finalize();
            exit(0);
        }
        else if (arg == "--train")
        {
            cfg.train_path = need_value(arg);
        }
        else if (arg == "--test")
        {
            cfg.test_path = need_value(arg);
        }
        else if (arg == "--test-limit")
        {
            cfg.test_limit = stoi(need_value(arg));
        }
        else if (arg == "--max-guesses")
        {
            cfg.max_guesses = stoll(need_value(arg));
        }
        else if (arg == "--hash-threshold")
        {
            cfg.hash_threshold = stoi(need_value(arg));
        }
        else if (arg == "--batch-size")
        {
            cfg.batch_size = stoi(need_value(arg));
        }
        else if (arg == "--local-mode")
        {
            cfg.local_mode = need_value(arg);
        }
        else if (arg == "--local-threads")
        {
            cfg.local_threads = stoi(need_value(arg));
        }
        else if (arg == "--hash-mode")
        {
            cfg.hash_mode = need_value(arg);
        }
        else if (arg == "--pt-mpi")
        {
            cfg.enable_pt_level_mpi = is_true_flag(need_value(arg));
        }
        else if (rank == 0)
        {
            cerr << "Warning: unknown option ignored: " << arg << endl;
        }
    }

    if (cfg.test_path.empty())
    {
        cfg.test_path = cfg.train_path;
    }
    cfg.test_limit = max(1, cfg.test_limit);
    cfg.hash_threshold = max(1, cfg.hash_threshold);
    cfg.local_threads = max(1, cfg.local_threads);
    return cfg;
}

static unordered_set<string> load_test_set(const string &path, int limit)
{
    unordered_set<string> test_set;
    ifstream test_data(path);
    string pw;
    int count = 0;

    while (test_data >> pw)
    {
        test_set.insert(pw);
        count++;
        if (count >= limit)
        {
            break;
        }
    }
    return test_set;
}

static bool can_simd4(const string inputs[4])
{
    size_t len = inputs[0].size();
    for (int i = 1; i < 4; i++)
    {
        if (inputs[i].size() != len)
        {
            return false;
        }
    }

    // 当前 SIMD4 MD5 主要面向 Rockyou 常见短口令的单块 MD5。
    // 超长口令回退到标量版本，避免不同 padding 块数造成结果不一致。
    return len <= 55;
}

static void hash_batch_scalar(const vector<string> &batch,
                              const unordered_set<string> &test_set,
                              HashStats &stats)
{
    bit32 state[4];
    for (const string &pw : batch)
    {
        if (test_set.find(pw) != test_set.end())
        {
            stats.cracked++;
        }
        MD5Hash(pw, state);
        stats.hashed++;
    }
}

static void hash_batch_simd(const vector<string> &batch,
                            const unordered_set<string> &test_set,
                            HashStats &stats)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    size_t i = 0;
    for (; i + 3 < batch.size(); i += 4)
    {
        string inputs[4] = {batch[i], batch[i + 1], batch[i + 2], batch[i + 3]};
        if (!can_simd4(inputs))
        {
            vector<string> fallback = {batch[i], batch[i + 1], batch[i + 2], batch[i + 3]};
            hash_batch_scalar(fallback, test_set, stats);
            continue;
        }

        for (int k = 0; k < 4; k++)
        {
            if (test_set.find(inputs[k]) != test_set.end())
            {
                stats.cracked++;
            }
        }

        bit32 states[4][4];
        MD5Hash_SIMD4(inputs, states);
        stats.hashed += 4;
    }

    if (i < batch.size())
    {
        vector<string> tail(batch.begin() + i, batch.end());
        hash_batch_scalar(tail, test_set, stats);
    }
#else
    // 非 ARM NEON 平台上自动退化为标量哈希，保证 general 分支可移植。
    hash_batch_scalar(batch, test_set, stats);
#endif
}

static void push_guesses_to_hash_queue(vector<string> &guesses)
{
    if (guesses.empty())
    {
        return;
    }

    vector<string> batch;
    batch.swap(guesses);

    {
        lock_guard<mutex> lock(hash_mutex);
        pending_hash_batches.emplace_back(move(batch));
    }
    hash_cv.notify_one();
}

static void hash_worker_thread(const unordered_set<string> &test_set,
                               const string &hash_mode,
                               HashStats &stats)
{
    while (true)
    {
        vector<string> local_batch;
        {
            unique_lock<mutex> lock(hash_mutex);
            hash_cv.wait(lock, [] {
                return hash_thread_should_exit.load() || !pending_hash_batches.empty();
            });

            if (pending_hash_batches.empty() && hash_thread_should_exit.load())
            {
                break;
            }

            local_batch = move(pending_hash_batches.front());
            pending_hash_batches.pop_front();
        }

        auto start_hash = system_clock::now();
        if (hash_mode == "simd" || hash_mode == "SIMD")
        {
            hash_batch_simd(local_batch, test_set, stats);
        }
        else
        {
            hash_batch_scalar(local_batch, test_set, stats);
        }
        auto end_hash = system_clock::now();
        auto duration_hash = duration_cast<microseconds>(end_hash - start_hash);
        stats.time_hash += double(duration_hash.count()) * microseconds::period::num / microseconds::period::den;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    GeneralConfig cfg = parse_args(argc, argv, rank);
    if (cfg.batch_size <= 0)
    {
        cfg.batch_size = max(1, size);
    }

    double time_guess = 0.0;
    double time_train = 0.0;
    double time_total = 0.0;
    long long local_generated_total = 0;

    auto start_total = system_clock::now();

    PriorityQueue q;
    q.mpi_rank = rank;
    q.mpi_size = size;
    q.SetLocalMode(cfg.local_mode, cfg.local_threads);

    auto start_train = system_clock::now();
    q.m.train(cfg.train_path);
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;

    unordered_set<string> test_set = load_test_set(cfg.test_path, cfg.test_limit);
    q.init();

    if (rank == 0)
    {
        cout << "Starting general fused version" << endl;
        cout << "MPI size: " << size << endl;
        cout << "PT-level MPI: " << (cfg.enable_pt_level_mpi ? "on" : "off") << endl;
        cout << "Local generation mode: " << cfg.local_mode << endl;
        cout << "Local threads: " << cfg.local_threads << endl;
        cout << "Hash mode: " << cfg.hash_mode << endl;
        cout << "Batch size: " << cfg.batch_size << endl;
        cout << "Hash threshold: " << cfg.hash_threshold << endl;
        cout << "Max guesses: " << cfg.max_guesses << endl;
    }

    HashStats local_hash_stats;
    hash_thread_should_exit = false;
    thread hash_thread(hash_worker_thread, cref(test_set), cref(cfg.hash_mode), ref(local_hash_stats));

    long long window_generated = 0;
    long long history_generated = 0;
    long long next_report = 100000;
    bool should_continue = true;

    while (should_continue)
    {
        int local_has_work = q.priority.empty() ? 0 : 1;
        int global_has_work = 0;
        MPI_Allreduce(&local_has_work, &global_has_work, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_has_work == 0)
        {
            break;
        }

        if (!q.priority.empty())
        {
            auto start_guess = system_clock::now();
            q.PopNextHybrid(cfg.batch_size, cfg.enable_pt_level_mpi);
            auto end_guess = system_clock::now();
            auto duration_guess = duration_cast<microseconds>(end_guess - start_guess);
            time_guess += double(duration_guess.count()) * microseconds::period::num / microseconds::period::den;
        }

        long long local_added = (long long)q.last_generate_count;
        local_generated_total += local_added;

        long long global_added = 0;
        MPI_Allreduce(&local_added, &global_added, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        window_generated += global_added;

        if (rank == 0 && history_generated + window_generated >= next_report)
        {
            cout << "Guesses generated: " << history_generated + window_generated << endl;
            while (next_report <= history_generated + window_generated)
            {
                next_report += 100000;
            }
        }

        // 达到阈值后把本进程已经生成的候选交给哈希线程，使生成和哈希重叠执行。
        if (window_generated >= cfg.hash_threshold)
        {
            push_guesses_to_hash_queue(q.guesses);
            history_generated += window_generated;
            window_generated = 0;
        }

        int should_exit = 0;
        if (rank == 0 && history_generated + window_generated >= cfg.max_guesses)
        {
            should_exit = 1;
        }
        MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (should_exit)
        {
            push_guesses_to_hash_queue(q.guesses);
            should_continue = false;
        }
    }

    // 退出前处理尚未进入哈希队列的候选。
    push_guesses_to_hash_queue(q.guesses);

    hash_thread_should_exit = true;
    hash_cv.notify_all();
    hash_thread.join();

    auto end_total = system_clock::now();
    auto duration_total = duration_cast<microseconds>(end_total - start_total);
    time_total = double(duration_total.count()) * microseconds::period::num / microseconds::period::den;

    long long global_generated = 0;
    long long global_cracked = 0;
    long long global_hashed = 0;

    MPI_Reduce(&local_generated_total, &global_generated, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_stats.cracked, &global_cracked, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_stats.hashed, &global_hashed, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_guess_time = 0.0;
    double global_hash_time = 0.0;
    double global_train_time = 0.0;
    double global_total_time = 0.0;

    MPI_Reduce(&time_guess, &global_guess_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_hash_stats.time_hash, &global_hash_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_train, &global_train_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_total, &global_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "========== General Fused Result ==========" << endl;
        cout << "Generate time: " << global_guess_time << " seconds" << endl;
        cout << "Hash time: " << global_hash_time << " seconds" << endl;
        cout << "Train time: " << global_train_time << " seconds" << endl;
        cout << "Total time: " << global_total_time << " seconds" << endl;
        cout << "Generated: " << global_generated << endl;
        cout << "Hashed: " << global_hashed << endl;
        cout << "Cracked: " << global_cracked << endl;
        cout << "==========================================" << endl;
    }

    MPI_Finalize();
    return 0;
}
