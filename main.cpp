#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>

using namespace std;

const int MAX_USER_ID = 10005;

struct MessageTask {
    int msgType;
    int usrInst;
    int exeTime;
    int deadLine;
    bool isOverTime;

    MessageTask(int mt, int ui, int et, int dl) : msgType(mt), usrInst(ui), exeTime(et), deadLine(dl),
                                                  isOverTime(false) {}

    bool operator<(const MessageTask &other) const {
        return deadLine < other.deadLine;
    }
};

vector<vector<MessageTask>> usrInst2tasks(MAX_USER_ID);
vector<vector<MessageTask>> cores;
vector<int> usrInst2totalTime(MAX_USER_ID, 0);
vector<vector<int>> cores2usrInst;
int n, m, c;

int read() {
    int x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-')
            f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}

void get_input() {
    n = read(), m = read(), c = read();
    for (int i = 0; i < n; ++i) {
        int msgType, usrInst, exeTime, deadLine;
        msgType = read(), usrInst = read(), exeTime = read(), deadLine = read();
        deadLine = min(deadLine, c);
        usrInst2tasks[usrInst].emplace_back(msgType, usrInst, exeTime, deadLine);
    }
}

void output_result() {
    for (const auto &core_tasks: cores) {
        cout << core_tasks.size();
        for (const auto &task: core_tasks) {
            cout << " " << task.msgType << " " << task.usrInst;
        }
        cout << "\n";
    }
}

void pre_process() {
    for (int usrInst = 0; usrInst < MAX_USER_ID; ++usrInst) {
        int total_time = 0;
        for (auto &task: usrInst2tasks[usrInst]) {
            total_time += task.exeTime;
            if (total_time > task.deadLine) {
                task.isOverTime = true;
            }
        }
        usrInst2totalTime[usrInst] = total_time;
    }
}

void assign_core() {
    /**
     * 分配核心
     * 1. 按照用户实例的总执行时间从小到大排序
     * 2. 从小到大遍历用户实例，依次分配到核心上，尽量使得每个核心的总执行时间接近，如果将要超时则优寻找下一个核心
     */
    cores2usrInst.assign(m, vector<int>());
    vector<int> sorted_usrInst(MAX_USER_ID);
    iota(sorted_usrInst.begin(), sorted_usrInst.end(), 0);
    sort(sorted_usrInst.begin(), sorted_usrInst.end(),
         [](int a, int b) { return usrInst2totalTime[a] < usrInst2totalTime[b]; });

    int current_core = 0;
    vector<int> cores2totalTime(m, 0);

    for (int usrInst: sorted_usrInst) {
        if (usrInst2tasks[usrInst].empty())
            continue;
        for (int i = 0; i < m; ++i) {
            // 其实根据测试数据，没有出现超时的情况，不知道最后的数据会不会出现超时的情况
            if (cores2totalTime[current_core] > c) {
                current_core = (current_core + 1) % m;
                continue;
            }
            cores2usrInst[current_core].push_back(usrInst);
            cores2totalTime[current_core] += usrInst2totalTime[usrInst];
            current_core = (current_core + 1) % m;
            break;
        }
    }
}

struct TaskHeap {
    /**
     * 自定义堆，用于维护任务队列
     * 主要由于需要动态更新上一个任务的类型当前核的总执行时间，需要动态的cmp
     */

    vector<MessageTask> queue;
    int core_task_type;
    int core_time;
    bool need_re_heapify;

    TaskHeap(const vector<MessageTask> &tasks) : core_task_type(-1), core_time(-1), need_re_heapify(true) {
        for (const auto &task: tasks) {
            queue.emplace_back(task);
        }
        re_heapify();
    }

    bool cmp(const MessageTask &a, const MessageTask &b) {
        // 最后这里调整各个顺序尝试后提高了几万分

        // 优先级：1. 任务类型 2. 是否超时
        if (b.msgType == core_task_type)
            return false;
        if (a.msgType == core_task_type)
            return true;

        if (a.isOverTime)
            return false;

        if (b.isOverTime)
            return true;

        // 将要超时
        // 组员发现这里注释掉后分数会更高，总之因为没有新想法了所以在乱试
        /* if (a.deadLine <= core_time + a.exeTime)
            return true;
        if (b.deadLine <= core_time + b.exeTime)
            return false; */
        return a.deadLine < b.deadLine;
    }

    // 重新建堆
    void re_heapify() {
        if (!need_re_heapify)
            return;
        need_re_heapify = false;
        for (int i = queue.size() / 2; i >= 0; --i) {
            heapify_down(i);
        }
    }

    // 向上调整
    void heapify_up(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (cmp(queue[parent], queue[idx])) {
                break;
            }
            swap(queue[parent], queue[idx]);
            idx = parent;
        }
    }

    // 向下调整
    void heapify_down(int idx) {
        while (idx < queue.size()) {
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            int min_idx = idx;
            if (left < queue.size() && cmp(queue[left], queue[min_idx])) {
                min_idx = left;
            }
            if (right < queue.size() && cmp(queue[right], queue[min_idx])) {
                min_idx = right;
            }
            if (min_idx == idx) {
                break;
            }
            swap(queue[min_idx], queue[idx]);
            idx = min_idx;
        }
    }

    void push(const MessageTask &task) {
        queue.emplace_back(task);
        heapify_up(queue.size() - 1);
    }

    MessageTask pop() {
        MessageTask task = queue[0];
        queue[0] = queue.back();
        queue.pop_back();
        heapify_down(0);
        return task;
    }

    bool empty() const {
        return queue.empty();
    }

    void change_core_task_type(int task_type) {
        if (core_task_type != task_type)
            need_re_heapify = true;
        core_task_type = task_type;
    }

    void change_core_time(int time) {
        // 这里没加判断是怕超时，同时总执行时间是每次一定会变的，从而每次都要重新建堆
        // 一开始是因为超时没加，后来忘了这块了，可能加上后分数会更高
        core_time = time;
    }
};

void assign_tasks() {
    cores.assign(m, vector<MessageTask>());
    for (int coreId = 0; coreId < m; ++coreId) {
        int core_time = 0;
        TaskHeap possible_tasks(vector<MessageTask>{});
        for (int usrInst: cores2usrInst[coreId]) {
            if (!usrInst2tasks[usrInst].empty()) {
                possible_tasks.push(usrInst2tasks[usrInst][0]);
            }
        }
        while (!possible_tasks.empty()) {
            auto task = possible_tasks.pop();
            cores[coreId].push_back(task);
            core_time += task.exeTime;
            int usrInst = task.usrInst;
            usrInst2tasks[usrInst].erase(usrInst2tasks[usrInst].begin());
            if (!usrInst2tasks[usrInst].empty()) {
                possible_tasks.push(usrInst2tasks[usrInst][0]);
            }
            possible_tasks.change_core_task_type(task.msgType);
            possible_tasks.change_core_time(core_time);
            possible_tasks.re_heapify();
        }
    }
}

void main_task() {
    // 读取输入
    get_input();
    // 预处理，计算每个用户实例的总执行时间和是否一定超时
    pre_process();
    // 分配核心，将用户实例分配到核心上
    assign_core();
    // 分配任务，将任务分配到核心上
    assign_tasks();
    // 输出结果
    output_result();
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    main_task();
    return 0;
}
