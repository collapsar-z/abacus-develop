# HOWTO: 在新分支重新加入 DiagoAutoSelector

这份文档写给后续维护者或 Codex，用来把“PW 迭代对角化 AutoSelector”迁移到一个新的 ABACUS 分支上。它不是展示报告，而是实施指南。

目标：在 PW hsolver 中增加一个可解释的求解器推荐层。默认只报告推荐，不改变用户指定的 `ks_solver`；只有显式设置环境变量时才自动切换。

## 1. 功能边界

AutoSelector 第一版只做 PW 迭代求解器推荐：

- `cg`
- `dav`
- `dav_subspace`
- `bpcg`
- `ppcg`

不要在第一版里做这些事：

- 不要修改输入文件格式。
- 不要强制新增 `ks_solver=auto`。
- 不要默认改变用户选择。
- 不要处理 LCAO 的 LAPACK/ELPA/ScaLAPACK 路径。
- 不要把推荐规则写散在 `hsolver_pw.cpp` 里。

两个环境变量：

```bash
ABACUS_DIAGO_AUTO_REPORT=1  # 只输出推荐，不改变实际 solver
ABACUS_DIAGO_AUTO_SELECT=1  # 使用推荐 solver 覆盖当前 method
```

## 2. 新增文件

新增：

```text
source/source_hsolver/module_diag/diago_auto_selector.h
```

参考实现：

```cpp
#ifndef DIAGO_AUTO_SELECTOR_H_
#define DIAGO_AUTO_SELECTOR_H_

#include <cstdlib>
#include <sstream>
#include <string>

namespace hsolver
{

struct DiagoAutoSelectInput
{
    std::string current_method;
    std::string calculation;
    int nbands = 0;
    int nbasis = 0;
    int npw_total = 0;
    int nproc_in_pool = 1;
    int scf_iter = 1;
    bool gpu_device = false;
};

struct DiagoAutoSelectResult
{
    std::string method;
    std::string reason;
};

class DiagoAutoSelector
{
  public:
    static bool report_enabled()
    {
        return env_enabled("ABACUS_DIAGO_AUTO_REPORT") || auto_select_enabled();
    }

    static bool auto_select_enabled()
    {
        return env_enabled("ABACUS_DIAGO_AUTO_SELECT");
    }

    static DiagoAutoSelectResult recommend_pw(const DiagoAutoSelectInput& input)
    {
        DiagoAutoSelectResult result;
        result.method = input.current_method;

        const int nbands = input.nbands > 0 ? input.nbands : 1;
        const double basis_per_band = static_cast<double>(input.npw_total > 0 ? input.npw_total : input.nbasis)
                                      / static_cast<double>(nbands);

        std::ostringstream reason;
        reason << "basis_per_band=" << basis_per_band
               << ", nbands=" << input.nbands
               << ", nproc_pool=" << input.nproc_in_pool
               << ", calculation=" << input.calculation
               << ", device=" << (input.gpu_device ? "GPU" : "CPU");

        if (input.gpu_device)
        {
            result.method = "bpcg";
            reason << "; recommend bpcg because block CG is the GPU-oriented iterative path";
        }
        else if (input.calculation == "nscf")
        {
            result.method = "dav";
            reason << "; recommend dav because nscf usually benefits from robust full convergence";
        }
        else if (input.nproc_in_pool > 1 && input.nbands >= 32)
        {
            result.method = "bpcg";
            reason << "; recommend bpcg because many bands with MPI can benefit from block operations";
        }
        else if (input.nbands >= 64)
        {
            result.method = "ppcg";
            reason << "; recommend ppcg because many bands make projected/block updates attractive";
        }
        else if (basis_per_band > 80.0 && input.scf_iter > 1)
        {
            result.method = "dav_subspace";
            reason << "; recommend dav_subspace for large PW subspaces after the initial SCF step";
        }
        else
        {
            result.method = "cg";
            reason << "; recommend cg as the conservative default for small or early PW solves";
        }

        result.reason = reason.str();
        return result;
    }

  private:
    static bool env_enabled(const char* name)
    {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }
};

} // namespace hsolver

#endif // DIAGO_AUTO_SELECTOR_H_
```

这是 header-only，不需要改 `CMakeLists.txt`。

## 3. 接入 HSolverPW

文件：

```text
source/source_hsolver/hsolver_pw.cpp
```

加入 include：

```cpp
#include "source_base/module_device/device.h"
#include "source_hsolver/module_diag/diago_auto_selector.h"
```

在 `HSolverPW<T, Device>::hamiltSolvePsiK(...)` 中，先找到这段逻辑：

```cpp
const int cur_nbasis = psi.get_current_nbas();
const int nbands = psi.get_nbands();
int npw_total = cur_nbasis;
#ifdef __MPI
...
#endif
```

这段通常在创建 `hpsi_func` 和 `spsi_func` 之前。AutoSelector 应该放在 rank deficiency 检查之后、solver 分支之前。

插入：

```cpp
std::string effective_method = this->method;
DiagoAutoSelectInput auto_input;
auto_input.current_method = this->method;
auto_input.calculation = this->calculation_type;
auto_input.nbands = nbands;
auto_input.nbasis = psi.get_nbasis();
auto_input.npw_total = npw_total;
auto_input.nproc_in_pool = this->nproc_in_pool;
auto_input.scf_iter = this->scf_iter;
auto_input.gpu_device = base_device::get_device_type(this->ctx) == base_device::GpuDevice;
const DiagoAutoSelectResult auto_result = DiagoAutoSelector::recommend_pw(auto_input);
if (DiagoAutoSelector::report_enabled() && GlobalV::MY_RANK == 0)
{
    GlobalV::ofs_running << "[DiagoAutoSelector] current=" << this->method
                         << " recommended=" << auto_result.method
                         << " reason: " << auto_result.reason << std::endl;
}
if (DiagoAutoSelector::auto_select_enabled())
{
    effective_method = auto_result.method;
}
```

然后把后面的 solver 分支从：

```cpp
if (this->method == "cg")
...
else if (this->method == "bpcg")
...
else if (this->method == "ppcg")
...
else if (this->method == "dav_subspace")
...
else if (this->method == "dav")
```

改成：

```cpp
if (effective_method == "cg")
...
else if (effective_method == "bpcg")
...
else if (effective_method == "ppcg")
...
else if (effective_method == "dav_subspace")
...
else if (effective_method == "dav")
```

不要改 `solve(...)` 开头的 supported methods 检查。它仍然检查用户输入的 `this->method` 是否属于支持列表。

## 4. 推荐规则的依据

这不是“最优求解器证明器”，而是启发式推荐器。文档和报告里要诚实表达。

当前规则依据：

| 场景 | 推荐 | 理由 |
| --- | --- | --- |
| GPU device | `bpcg` | block CG 路径更适合 GPU/块操作 |
| `nscf` | `dav` | 非自洽计算通常更强调稳健全收敛 |
| MPI pool + 多 band | `bpcg` | 多 band 和 MPI 更适合块并行 |
| band 数很多 | `ppcg` | projected/block update 更有价值 |
| 大 PW 子空间且不是初始 SCF | `dav_subspace` | 子空间方法可利用已有近似空间 |
| 小体系或默认情况 | `cg` | 简单保守，额外开销低 |

如果新分支已有更可靠的 benchmark 数据，可以调整阈值，例如：

```cpp
input.nbands >= 32
input.nbands >= 64
basis_per_band > 80.0
```

但不要把阈值写得太复杂。第一版重点是“可解释”。

## 5. 验证方式

默认模式下应完全不改变行为：

```bash
unset ABACUS_DIAGO_AUTO_REPORT
unset ABACUS_DIAGO_AUTO_SELECT
cmake --build build-orth --target MODULE_HSOLVER_pw -j$(nproc)
ctest --test-dir build-orth -R "MODULE_HSOLVER_pw" --output-on-failure
```

只报告推荐：

```bash
export ABACUS_DIAGO_AUTO_REPORT=1
unset ABACUS_DIAGO_AUTO_SELECT
ctest --test-dir build-orth -R "MODULE_HSOLVER_pw" --output-on-failure
```

检查 running log 中是否出现：

```text
[DiagoAutoSelector] current=... recommended=... reason: ...
```

自动切换模式需要谨慎测试：

```bash
export ABACUS_DIAGO_AUTO_SELECT=1
ctest --test-dir build-orth -R "MODULE_HSOLVER_pw" --output-on-failure
```

如果自动切换导致某些测试失败，先保留 report-only 模式，不要强行启用 auto select。

## 6. 常见坑

- 不要默认开启自动切换。默认只能保持原行为。
- `ABACUS_DIAGO_AUTO_REPORT=1` 应只输出推荐，不改变实际 solver。
- `ABACUS_DIAGO_AUTO_SELECT=1` 才允许改 `effective_method`。
- 只在 `GlobalV::MY_RANK == 0` 输出推荐，避免 MPI 多 rank 刷屏。
- 不要把 LCAO solver 混进 PW AutoSelector。
- 如果新分支已经改了 `HSolverPW::hamiltSolvePsiK` 结构，按语义找位置：rank deficiency 检查之后，`if solver == ...` 分支之前。
- 如果新分支没有 `base_device::get_device_type(this->ctx)`，先查已有代码如何判断 GPU device，不要硬写。

## 7. 与 DiagoTrace 的关系

AutoSelector 可以单独存在，但最好和 DiagoTrace 一起使用：

```bash
export ABACUS_DIAGO_AUTO_REPORT=1
export ABACUS_DIAGO_TRACE=1
export ABACUS_DIAGO_TRACE_FILE=trace.csv
```

AutoSelector 负责给出推荐和理由；DiagoTrace 负责记录真实收敛过程。外部 benchmark 可以用总时间验证推荐是否合理，用 trace CSV 分析为什么某个 solver 更快或更慢。
