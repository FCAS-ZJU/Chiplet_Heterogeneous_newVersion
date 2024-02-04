# Chiplet_Heterogeneous_newVersion

本版本仿真器使用了GPGPU-Sim 4.0.0版本
本仓库的说明文档已上传，仿真器的安装、使用请阅读“基于多芯粒集成的CPU-GPU异构的消息传递式仿真器说明文档v1.3.pdf”


## 仓库调整说明
（适当时间删除）

针对仓库特性，进行了如下调整：

1. 将示例程序matmul提取到benchmark/matmul，并提供编译和运行的makefile。
2. 将自研的API和通信函数提取到interchiplet，并修改了对应的CMakeList。
   1. 此文件夹下面包含record_transfer、libinterchiplet_app.a
3. 将snipersim和gpgpu-sim替换为submodule，减小仓库规模。
4. 将snipersim和gpgpu-sim中需要修改的文件以patch形式提取到interchiplet/patch中。提供应用patch的访问修改。
   1. 提供了apply_patch.sh脚本将修改应用到snipersim和gpgpu-sim
   2. 提供了patch.sh脚本将snipersim和gpgpu-sim的修改压缩到patch，并复制到.changed_files文件夹作为备份。
5. 提供setup_env.sh脚本自动化环境设置。

下面介绍新的安装流程。


## 安装

### 下载仓库并设置环境

1. 从github上下载仓库。

    ```
    git clone https://github.com/FCAS-SCUT/Chiplet_Heterogeneous_newVersion.git
    ```

    进入仿真器根目录，以下的示例命名都假设从仿真器根目录开始执行。

2. 初始化并更新submodule。

    ```
    git submodule init
    git submodule update
    ```

3. 运行脚本，初始化环境变量

    ```
    source setup_env.sh
    ```

    运行成功应出现：setup_environment succeeded

4. 对于snipersim和gpgpu-sim代码进行修改。

    ```
    ./apply_patch.sh
    ```

    更多细节参见下文“打包和应用Patch”章节。

5. 编译安装snipersim。新版本的snipersim提供了非常自动化的编译脚本，直接执行make即可。

    ```
    cd snipersim
    make -j4
    ```

6. 编译安装GPGPUSim。GPGPUsim安装有前置条件：

    1. GPGPUSim需要安装cuda。新版本的gpgpusim可以支持cuda4到cuda11的任意版本，详细信息请参见GPGPUSim的README。
    2. GPGPUSim对于编译版本有要求，建议使用GCC7。

    配置好Cuda和编译器，可以直接执行make。

    ```
    cd gpgpu-sim
    make -j4
    ```

7. 编译安装popnet

    ```
    cd popnet
    make -j4
    ```

8. 编译安装芯粒间通信程序。interchiplet提供了芯粒间通信所需要的API和实现代码。

    ```
    cd interchiplet
    mkdir build
    cd build
    cmake ..
    make
    ```

    编译完成后应在interchiplet/bin下找到record_transfer和zmq_pro，在interchiplet/lib下找到libinterchiplet_app.a。

    zmq_pro需要安装zmq环境。通常会在cmake步骤被忽略。

## 验证安装

正确执行上述过程后，可以使用benchmark/matmul验证环境设置是否正确。

1. 设置仿真器环境

    ```
    source setup_env.sh
     ```

2. 编译可执行文件

    ```
    cd benchmark/matmul
    make
    ```

3. 执行可执行文件。示例包含4个进程，分别是1个CPU进行和3个GPU进程。

    ```
    make run
    ```

    执行后，可以在benchmark/matmul文件下找到如下文件：

    1. 通信记录文件bench.0.0.txt和通信数据文件。
    2. GPGPUSim仿真的日志文件gpgpusim_X_X.log。
    3. GPGPUSim仿真的临时文件。
    4. Sniper仿真的临时文件。

4. 清理仿真的输出文件。

    ```
    make clean
    make cleanall # 同时清理可执行文件
    ```

## 打包和应用Patch

由于sniper和GPGPUSim是用submodule方式引入的，对于snipersim和gpgpu-sim的修改不会通过常规的git流程追踪。因此，工程提供了patch.sh和apply_patch.sh两个脚本通过Patch管理sniper和gpgpu-sim的修改。

patch.sh脚本用来生成Patch：

```
./patch.sh
```

1. 使用patch.sh脚本将snipersim和gpgpu-sim的修改分别打包到snipersim.diff和gpgpu-sim.diff文件中。diff文件保存在interchiplet/patch下面。diff文件会被git追踪。
2. patch.sh脚本还会将被修改的文件按照文件层次结构保存到.changed_files文件夹中，用于在diff文件出错时进行查看和参考。

apply_patch.sh脚本用来应用Patch：

```
./apply_patch.sh
```

1. 使用apply_patch.sh脚本将snipersim.diff和gpgpu-sim.diff文件应用到snipersim和gpgpu-sim，重现对于文件的修改。
2. 当apply出错时，可以参考.changed_files中的文件手动修改snipersim和gpgpu-sim的文件。

需要说明的是：不建议用.changed_files直接覆盖snipersim和gpgpu-sim文件夹。因为snipersim和gpgpu-sim本身的演进可能会与芯粒仿真器修改相同的文件。使用Patch的方式会报告修改的冲突。如果直接覆盖，则会导致不可预见的错误。
