#include "PCFG.h"

#include <algorithm>
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

namespace
{
struct PThreadGenerateTask
{
    PriorityQueue *queue;
    PT pt;
    int start_idx;
    int end_idx;
    vector<string> *output;
};

void *PThreadGenerateWorker(void *arg)
{
    PThreadGenerateTask *task = static_cast<PThreadGenerateTask *>(arg);
    task->queue->GenerateRange(task->pt, task->start_idx, task->end_idx, *task->output);
    return nullptr;
}
} // namespace

void PriorityQueue::SetLocalMode(const string &mode_name, int threads)
{
    if (mode_name == "openmp" || mode_name == "omp" || mode_name == "OPENMP")
    {
        local_mode = LocalGenerateMode::OPENMP;
    }
    else if (mode_name == "pthread" || mode_name == "pthreads" || mode_name == "PTHREAD")
    {
        local_mode = LocalGenerateMode::PTHREAD;
    }
    else
    {
        local_mode = LocalGenerateMode::SERIAL;
    }

    local_threads = max(1, threads);
}

segment *PriorityQueue::GetSegmentTable(segment seg)
{
    if (seg.type == 1)
    {
        return &m.letters[m.FindLetter(seg)];
    }
    if (seg.type == 2)
    {
        return &m.digits[m.FindDigit(seg)];
    }
    if (seg.type == 3)
    {
        return &m.symbols[m.FindSymbol(seg)];
    }
    return nullptr;
}

int PriorityQueue::GetLastSegmentSize(PT pt)
{
    if (pt.content.empty())
    {
        return 0;
    }

    int last_idx = (int)pt.content.size() - 1;
    if (last_idx < (int)pt.max_indices.size())
    {
        return pt.max_indices[last_idx];
    }

    segment *table = GetSegmentTable(pt.content[last_idx]);
    return table == nullptr ? 0 : (int)table->ordered_values.size();
}

string PriorityQueue::BuildPrefix(PT pt)
{
    string prefix;
    if (pt.content.size() <= 1)
    {
        return prefix;
    }

    for (int seg_idx = 0; seg_idx < (int)pt.content.size() - 1; seg_idx++)
    {
        segment *table = GetSegmentTable(pt.content[seg_idx]);
        if (table == nullptr)
        {
            continue;
        }

        if (seg_idx < (int)pt.curr_indices.size())
        {
            int value_idx = pt.curr_indices[seg_idx];
            if (value_idx >= 0 && value_idx < (int)table->ordered_values.size())
            {
                prefix += table->ordered_values[value_idx];
            }
        }
    }
    return prefix;
}

void PriorityQueue::AppendGenerated(vector<string> &local_guesses)
{
    if (local_guesses.empty())
    {
        return;
    }

    last_generate_count += local_guesses.size();
    total_guesses += (int)local_guesses.size();
    guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
    local_guesses.clear();
}

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (index >= (int)pt.content.size())
        {
            break;
        }

        if (pt.content[index].type == 1)
        {
            int seg_id = m.FindLetter(pt.content[index]);
            pt.prob *= m.letters[seg_id].ordered_freqs[idx];
            pt.prob /= m.letters[seg_id].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            int seg_id = m.FindDigit(pt.content[index]);
            pt.prob *= m.digits[seg_id].ordered_freqs[idx];
            pt.prob /= m.digits[seg_id].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            int seg_id = m.FindSymbol(pt.content[index]);
            pt.prob *= m.symbols[seg_id].ordered_freqs[idx];
            pt.prob /= m.symbols[seg_id].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    priority.clear();

    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back((int)m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back((int)m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back((int)m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }

        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }

    stable_sort(priority.begin(), priority.end(), [](const PT &a, const PT &b) {
        return a.prob > b.prob;
    });
}

void PriorityQueue::InsertNewPTs(const vector<PT> &new_pts)
{
    if (new_pts.empty())
    {
        return;
    }

    priority.insert(priority.end(), new_pts.begin(), new_pts.end());
    stable_sort(priority.begin(), priority.end(), [](const PT &a, const PT &b) {
        return a.prob > b.prob;
    });
}

void PriorityQueue::PopNext()
{
    if (priority.empty())
    {
        last_generate_count = 0;
        return;
    }

    last_generate_count = 0;
    PT current = priority.front();

    GenerateMPI(current);

    vector<PT> new_pts = current.NewPTs();
    for (PT &pt : new_pts)
    {
        CalProb(pt);
    }

    priority.erase(priority.begin());
    InsertNewPTs(new_pts);
}

void PriorityQueue::PopNextBatch(int batch_size)
{
    PopNextHybrid(batch_size, true);
}

void PriorityQueue::PopNextHybrid(int batch_size, bool enable_pt_level_mpi)
{
    if (priority.empty())
    {
        last_generate_count = 0;
        return;
    }

    last_generate_count = 0;
    int actual_batch_size = min(max(1, batch_size), (int)priority.size());

    vector<PT> batch_pts;
    batch_pts.reserve(actual_batch_size);
    for (int i = 0; i < actual_batch_size; i++)
    {
        batch_pts.emplace_back(priority[i]);
    }

    // 第一层融合：MPI 负责 PT 级任务划分。
    // 第二层融合：每个 rank 内部由 GenerateHybrid 决定使用串行、OpenMP 或 pthread。
    for (int i = 0; i < actual_batch_size; i++)
    {
        bool this_rank_should_generate = true;
        if (enable_pt_level_mpi && mpi_size > 1)
        {
            this_rank_should_generate = (i % mpi_size == mpi_rank);
        }

        if (this_rank_should_generate)
        {
            GenerateHybrid(batch_pts[i]);
        }
    }

    // 所有进程都按照同样的 batch_pts 推导新 PT，保证优先队列状态一致。
    vector<PT> all_new_pts;
    for (int i = 0; i < actual_batch_size; i++)
    {
        vector<PT> new_pts = batch_pts[i].NewPTs();
        for (PT &pt : new_pts)
        {
            CalProb(pt);
            all_new_pts.emplace_back(pt);
        }
    }

    priority.erase(priority.begin(), priority.begin() + actual_batch_size);
    InsertNewPTs(all_new_pts);
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
    {
        return res;
    }

    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i += 1)
    {
        curr_indices[i] += 1;
        if (curr_indices[i] < max_indices[i])
        {
            pivot = i;
            res.emplace_back(*this);
        }
        curr_indices[i] -= 1;
    }
    pivot = init_pivot;
    return res;
}

void PriorityQueue::GenerateRange(PT pt, int start_idx, int end_idx, vector<string> &out)
{
    if (pt.content.empty())
    {
        return;
    }

    CalProb(pt);
    int total_values = GetLastSegmentSize(pt);
    start_idx = max(0, start_idx);
    end_idx = min(end_idx, total_values);
    if (start_idx >= end_idx)
    {
        return;
    }

    int last_idx = (int)pt.content.size() - 1;
    segment *last_table = GetSegmentTable(pt.content[last_idx]);
    if (last_table == nullptr)
    {
        return;
    }

    string prefix = BuildPrefix(pt);
    out.reserve(out.size() + (end_idx - start_idx));
    for (int i = start_idx; i < end_idx; i++)
    {
        out.emplace_back(prefix + last_table->ordered_values[i]);
    }
}

void PriorityQueue::GenerateHybridRange(PT pt, int start_idx, int end_idx)
{
    int total_values = GetLastSegmentSize(pt);
    start_idx = max(0, start_idx);
    end_idx = min(end_idx, total_values);
    if (start_idx >= end_idx)
    {
        return;
    }

    if (local_mode == LocalGenerateMode::SERIAL || local_threads <= 1)
    {
        vector<string> local_guesses;
        GenerateRange(pt, start_idx, end_idx, local_guesses);
        AppendGenerated(local_guesses);
        return;
    }

    if (local_mode == LocalGenerateMode::OPENMP)
    {
#ifdef _OPENMP
        int last_idx = (int)pt.content.size() - 1;
        segment *last_table = GetSegmentTable(pt.content[last_idx]);
        if (last_table == nullptr)
        {
            return;
        }

        string prefix = BuildPrefix(pt);
        int local_count = end_idx - start_idx;
        vector<string> local_guesses(local_count);

#pragma omp parallel for num_threads(local_threads) schedule(static)
        for (int offset = 0; offset < local_count; offset++)
        {
            int value_idx = start_idx + offset;
            local_guesses[offset] = prefix + last_table->ordered_values[value_idx];
        }

        AppendGenerated(local_guesses);
        return;
#else
        vector<string> local_guesses;
        GenerateRange(pt, start_idx, end_idx, local_guesses);
        AppendGenerated(local_guesses);
        return;
#endif
    }

    if (local_mode == LocalGenerateMode::PTHREAD)
    {
        int local_count = end_idx - start_idx;
        int thread_count = min(local_threads, local_count);
        if (thread_count <= 1)
        {
            vector<string> local_guesses;
            GenerateRange(pt, start_idx, end_idx, local_guesses);
            AppendGenerated(local_guesses);
            return;
        }

        vector<pthread_t> tids(thread_count);
        vector<PThreadGenerateTask> tasks(thread_count);
        vector<vector<string>> parts(thread_count);

        bool pthread_ok = true;
        int base = local_count / thread_count;
        int rem = local_count % thread_count;
        int cursor = start_idx;

        for (int t = 0; t < thread_count; t++)
        {
            int take = base + (t < rem ? 1 : 0);
            tasks[t].queue = this;
            tasks[t].pt = pt;
            tasks[t].start_idx = cursor;
            tasks[t].end_idx = cursor + take;
            tasks[t].output = &parts[t];
            cursor += take;

            if (pthread_create(&tids[t], nullptr, PThreadGenerateWorker, &tasks[t]) != 0)
            {
                pthread_ok = false;
                break;
            }
        }

        if (!pthread_ok)
        {
            // 创建线程失败时回退到串行路径，保证结果正确。
            vector<string> local_guesses;
            GenerateRange(pt, start_idx, end_idx, local_guesses);
            AppendGenerated(local_guesses);
            return;
        }

        for (int t = 0; t < thread_count; t++)
        {
            pthread_join(tids[t], nullptr);
        }

        vector<string> merged;
        for (int t = 0; t < thread_count; t++)
        {
            merged.insert(merged.end(), parts[t].begin(), parts[t].end());
        }
        AppendGenerated(merged);
        return;
    }

    vector<string> local_guesses;
    GenerateRange(pt, start_idx, end_idx, local_guesses);
    AppendGenerated(local_guesses);
}

void PriorityQueue::GenerateHybrid(PT pt)
{
    GenerateHybridRange(pt, 0, GetLastSegmentSize(pt));
}

void PriorityQueue::Generate(PT pt)
{
    LocalGenerateMode old_mode = local_mode;
    int old_threads = local_threads;
    local_mode = LocalGenerateMode::SERIAL;
    local_threads = 1;
    GenerateHybrid(pt);
    local_mode = old_mode;
    local_threads = old_threads;
}

void PriorityQueue::GenerateMPI(PT pt)
{
    int total_values = GetLastSegmentSize(pt);
    if (total_values <= 0)
    {
        return;
    }

    if (mpi_size <= 1)
    {
        GenerateHybridRange(pt, 0, total_values);
        return;
    }

    int values_per_process = total_values / mpi_size;
    int remainder = total_values % mpi_size;
    int start_idx = mpi_rank * values_per_process + min(mpi_rank, remainder);
    int end_idx = start_idx + values_per_process + (mpi_rank < remainder ? 1 : 0);

    GenerateHybridRange(pt, start_idx, end_idx);
}
