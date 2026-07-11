#ifndef PCFG_H
#define PCFG_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <queue>
#include <vector>

using namespace std;

// 进程内部的候选口令生成方式。
// SERIAL：保留原始串行生成；OPENMP：用 OpenMP 并行枚举最后一个 segment；
// PTHREAD：用 pthread 手动划分最后一个 segment。
enum class LocalGenerateMode
{
    SERIAL,
    OPENMP,
    PTHREAD
};

class segment
{
public:
    int type;   // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如 S6 的长度就是 6

    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    void PrintSeg();

    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values;
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();
};

class PT
{
public:
    vector<segment> content;
    int pivot = 0;

    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();

    vector<int> curr_indices;
    vector<int> max_indices;
    float preterm_prob;
    float prob;
};

class model
{
public:
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;

    int GetNextPretermID()
    {
        preterm_id++;
        return preterm_id;
    };

    int GetNextLettersID()
    {
        letters_id++;
        return letters_id;
    };

    int GetNextDigitsID()
    {
        digits_id++;
        return digits_id;
    };

    int GetNextSymbolsID()
    {
        symbols_id++;
        return symbols_id;
    };

    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;
    vector<PT> ordered_pts;

    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
};

class PriorityQueue
{
public:
    vector<PT> priority;
    model m;

    // MPI 进程信息。mpi_size=1 时会自动退化为单进程。
    int mpi_rank = 0;
    int mpi_size = 1;

    // 进程内部并行方式。用于把 MPI 分到本进程的 PT 继续交给 OpenMP/pthread 处理。
    LocalGenerateMode local_mode = LocalGenerateMode::SERIAL;
    int local_threads = 1;

    // 最近一次 PopNext/PopNextBatch/PopNextHybrid 新增的本地候选数。
    // 这个字段用于流水线主程序统计增量，避免反复累计 q.guesses.size()。
    size_t last_generate_count = 0;

    void SetLocalMode(const string &mode_name, int threads);

    void CalProb(PT &pt);
    void init();

    // 原始串行生成：当前进程完整生成一个 PT。
    void Generate(PT pt);

    // 基础 MPI 生成：单个 PT 内部按照 rank/size 切分最后一个 segment。
    void GenerateMPI(PT pt);

    // general 融合版生成：由 local_mode 决定进程内部使用串行、OpenMP 或 pthread。
    void GenerateHybrid(PT pt);
    void GenerateHybridRange(PT pt, int start_idx, int end_idx);

    // 只生成指定区间 [start_idx, end_idx)，结果写入 out。该函数不直接修改 guesses。
    void GenerateRange(PT pt, int start_idx, int end_idx, vector<string> &out);

    // 基础 MPI 版本：每次处理一个 PT。
    void PopNext();

    // PT 层面并行版本：一次取出多个 PT，并按 i % mpi_size 分配给不同 rank。
    void PopNextBatch(int batch_size = 4);

    // general 融合版：PT 级 MPI + 进程内部 OpenMP/pthread/SERIAL。
    void PopNextHybrid(int batch_size = 4, bool enable_pt_level_mpi = true);

    // 将新 PT 按概率插回优先队列。
    void InsertNewPTs(const vector<PT> &new_pts);

    int total_guesses = 0;
    vector<string> guesses;

private:
    segment *GetSegmentTable(segment seg);
    int GetLastSegmentSize(PT pt);
    string BuildPrefix(PT pt);
    void AppendGenerated(vector<string> &local_guesses);
};

#endif
