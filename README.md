# zhilian-cpp
 第一届“华为智联杯”无线程序设计大赛——软件挑战赛道二等奖源码

队名：CUDA_out_of_memory（虽然我们一次都没遇到过

[赛事介绍](https://competition.huaweicloud.com/information/1000042059/introduction)

最终得分：12822999，排名第六

## 代码结构

调度算法在main.cpp

AI模型在model文件夹下，训练太烂的，最后准确率也就个68%，在AI方面没啥经验，都是临时现学的，要是准确率高点排名应该还能在上去

## 调度算法思路

主要部分是贪心，先做了预处理，计算每个用户对应的任务时间总和并计算该任务是否一定会超时（因为用户的任务有顺序要求，所以会出现某个用户的任务的ddl一定完不成）

接着分两步完成调度，先分配核对用户的对应关系，尽量平均分配使得每个核的总时间接近防止超时任务过多，其实就是根据用户对应的任务时间和排序，挨个对核进行分配

然后是对核上任务的分配，这里是耗时比较多部分，原来算法的版本是python，在线下比赛环节老是超时，后来我让chatGPT改成了C++版本，然后做了若干优化才最后跑出来的（直接突飞猛进到第四），主要思路是维护前一个核心任务类型和当前核心的总执行时间，对于所有用户的任务队列的第一个任务，优先取核心任务类型相同的和不超时的，原本用的是排序，复杂度过大容易超时，而后改成了最大堆，由于核心任务类型和总执行时间的动态的，所以是自定义的堆

最后就是输出，没什么好讲的

## 复盘

写这些的东西的时候仔细看了看代码，发现其实分配任务的时候当前核心的总执行时间其实没咋用上，因为核心任务类型不一定会变，而执行时间一定会变，所以每次都得重新建堆，一开始的python版本一直超时，所以就没加每次建堆，最后微调的时候发现不用这个反而高了（不知道为什么，有种AI炼丹的美），如果用上可能还能高点，也有可能超时，但最后一个小时忘记这块了所以没试

总之最后能拿个二等奖还是挺开心的，毕竟我们队对这类比赛都没啥经验，AI训练更是全部现学的（看了看那个98.4%的大佬发现我的方向好多都是对的，PCA、CNN什么的，可惜不会写导致最后还是用的MLP），最后能有这个成绩很满足了

## 附录

AI赛道第一名（98.4%准确率）：[Wireless-program](https://github.com/aqizhoua/Wireless-program)

调度算法第一名：[-](https://github.com/lxj2389287408/-)
