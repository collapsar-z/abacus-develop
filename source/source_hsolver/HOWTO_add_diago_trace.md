# HOWTO: 在新分支重新加入 DiagoTrace 收敛诊断

这份文档写给后续维护者或 Codex，用来把“迭代对角化收敛诊断 Trace”迁移到一个新的 ABACUS 分支上。它不是报告，而是实施指南。

目标：在不改变任何 `diag(...)` 对外接口、不影响默认 benchmark 的前提下，为 hsolver 的迭代求解器增加可选 CSV trace。

## 1. 功能边界

DiagoTrace 做的事情很小：

- 默认关闭。
- 只有设置环境变量 `ABACUS_DIAGO_TRACE=1` 时才写 CSV。
- 可通过 `ABACUS_DIAGO_TRACE_FILE=xxx.csv` 指定输出文件。
- 输出每轮迭代的收敛过程信息。

不要在这一功能里做这些事：

- 不要改输入文件格式。
- 不要改 `diag(...)` 函数签名。
- 不要默认打印额外日志。
- 不要默认影响 benchmark。
- 不要在 trace 里引入 MPI 通信策略变化。

## 2. 新增文件

新增：

```text
source/source_hsolver/module_diag/diago_trace.h
```

参考实现：

```cpp
#ifndef DIAGO_TRACE_H_
#define DIAGO_TRACE_H_

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>

namespace hsolver
{

class DiagoTrace
{
  public:
    explicit DiagoTrace(const std::string& solver_name) : enabled_(is_enabled())
    {
        if (!this->enabled_)
        {
            return;
        }

        const char* filename = std::getenv("ABACUS_DIAGO_TRACE_FILE");
        const std::string path = filename == nullptr || filename[0] == '\0'
                                     ? "diago_trace.csv"
                                     : std::string(filename);

        this->file_.open(path, std::ios::app);
        if (!this->file_)
        {
            this->enabled_ = false;
            return;
        }

        if (this->file_.tellp() == 0)
        {
            this->file_ << "solver,iter,nband,max_residual,avg_residual,n_converged,orth_error,note\n";
        }
        this->solver_name_ = solver_name;
    }

    bool enabled() const
    {
        return this->enabled_;
    }

    template <typename Real>
    void record_iteration(const int iter,
                          const int nband,
                          const Real max_residual,
                          const Real avg_residual,
                          const int n_converged,
                          const Real orth_error,
                          const std::string& note = "")
    {
        if (!this->enabled_)
        {
            return;
        }
        this->file_ << this->solver_name_ << ','
                    << iter << ','
                    << nband << ','
                    << std::setprecision(16) << max_residual << ','
                    << std::setprecision(16) << avg_residual << ','
                    << n_converged << ','
                    << std::setprecision(16) << orth_error << ','
                    << note << '\n';
        this->file_.flush();
    }

  private:
    static bool is_enabled()
    {
        const char* value = std::getenv("ABACUS_DIAGO_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }

    bool enabled_ = false;
    std::ofstream file_;
    std::string solver_name_;
};

} // namespace hsolver

#endif // DIAGO_TRACE_H_
```

这是 header-only，不需要改 `CMakeLists.txt`。

## 3. 接入 PPCG

文件：

```text
source/source_hsolver/diago_ppcg.cpp
```

加入 include：

```cpp
#include "source_hsolver/module_diag/diago_trace.h"
```

在 `DiagoPPCG<T, Device>::diag(...)` 开头，`timer::start` 后加入：

```cpp
DiagoTrace trace("PPCG");
```

在每轮 `calc_preconditioned_residual` 和 locked band 更新之后，`test_error` 之前加入：

```cpp
if (trace.enabled())
{
    Real max_residual = Real(0);
    Real avg_residual = Real(0);
    int n_converged = 0;
    for (int ib = 0; ib < this->n_band_l; ++ib)
    {
        max_residual = std::max(max_residual, this->h_err[ib]);
        avg_residual += this->h_err[ib];
        if (this->is_locked[ib])
        {
            ++n_converged;
        }
    }
    if (this->n_band_l > 0)
    {
        avg_residual /= this->n_band_l;
    }
    trace.record_iteration(iter,
                           this->n_band_l,
                           max_residual,
                           avg_residual,
                           n_converged,
                           Real(-1),
                           !this->block_sizes.empty() ? "blocked" : "");
}
```

说明：

- PPCG 的 residual 已经在 `h_err` host mirror 里。
- `is_locked` 可以作为已收敛 band 数。
- `orth_error` 先填 `-1`，除非新分支已经有便宜的正交性误差计算。

## 4. 接入 BPCG

文件：

```text
source/source_hsolver/diago_bpcg.cpp
```

加入 include：

```cpp
#include "source_hsolver/module_diag/diago_trace.h"

#include <algorithm>
#include <vector>
```

在 `DiagoBPCG<T, Device>::diag(...)` 中，确定 `max_iter` 后加入：

```cpp
DiagoTrace trace("BPCG");
```

在每轮 `calc_grad_with_block(...)` 后加入：

```cpp
if (trace.enabled())
{
    std::vector<Real> err_host(this->n_band_l);
    const Real* err_ptr = this->err_st.template data<Real>();
    if (this->err_st.device_type() == ct::DeviceType::GpuDevice)
    {
        syncmem_var_d2h_op()(err_host.data(), this->err_st.template data<Real>(), this->n_band_l);
        err_ptr = err_host.data();
    }

    Real max_residual = Real(0);
    Real avg_residual = Real(0);
    int n_converged = 0;
    for (int ib = 0; ib < this->n_band_l; ++ib)
    {
        max_residual = std::max(max_residual, err_ptr[ib]);
        avg_residual += err_ptr[ib];
        if (err_ptr[ib] <= ethr_band[ib])
        {
            ++n_converged;
        }
    }
    if (this->n_band_l > 0)
    {
        avg_residual /= this->n_band_l;
    }
    trace.record_iteration(ntry,
                           this->n_band_l,
                           max_residual,
                           avg_residual,
                           n_converged,
                           Real(-1));
}
```

说明：

- BPCG 的 `err_st` 可能在 GPU 上，所以 trace 开启时才做 host copy。
- 默认关闭时没有额外 copy。

## 5. 接入 CG

文件：

```text
source/source_hsolver/diago_cg.cpp
```

加入 include：

```cpp
#include <source_hsolver/module_diag/diago_trace.h>
#include <string>
```

在 `DiagoCG<T, Device>::diag_once(...)` 开头，`timer::start` 后加入：

```cpp
DiagoTrace trace("CG");
```

在内层 `do { ... } while` 循环中，`update_psi(...)` 后加入：

```cpp
trace.record_iteration(iter,
                       this->n_band_,
                       cg_norm,
                       cg_norm,
                       m + (converged ? 1 : 0),
                       Real(-1),
                       "band=" + std::to_string(m));
```

说明：

- CG 是 band-by-band 求解，不像 PPCG/BPCG 一次处理整个 block。
- 因此这里记录的是当前 band 的内层 `cg_norm`。
- `note` 字段记录 band 编号。

## 6. 接入 Davidson

文件：

```text
source/source_hsolver/diago_david.cpp
```

加入 include：

```cpp
#include "source_hsolver/module_diag/diago_trace.h"

#include <algorithm>
#include <string>
```

在 `DiagoDavid<T, Device>::diag_once(...)` 中，`timer::start("DiagoDavid", "diag_once")` 后加入：

```cpp
DiagoTrace trace("Davidson");
```

在 Davidson 每轮 `check_update` 中，原来一般有：

```cpp
this->notconv = 0;
for (int m = 0; m < nband; m++)
{
    convflag[m] = (std::abs(this->eigenvalue[m] - eigenvalue_in[m]) < ethr_band[m]);
    ...
}
```

改成记录 eigenvalue delta：

```cpp
this->notconv = 0;
Real max_delta = Real(0);
Real avg_delta = Real(0);
for (int m = 0; m < nband; m++)
{
    const Real delta = std::abs(this->eigenvalue[m] - eigenvalue_in[m]);
    max_delta = std::max(max_delta, delta);
    avg_delta += delta;
    convflag[m] = (delta < ethr_band[m]);
    if (!convflag[m])
    {
        unconv[this->notconv] = m;
        this->notconv++;
    }
    eigenvalue_in[m] = this->eigenvalue[m];
}
if (nband > 0)
{
    avg_delta /= nband;
}
trace.record_iteration(dav_iter,
                       nband,
                       max_delta,
                       avg_delta,
                       nband - this->notconv,
                       Real(-1),
                       "nbase=" + std::to_string(nbase));
```

说明：

- Davidson 这里没有直接 residual 数组。
- 它的收敛判断本来就是 eigenvalue 变化量，因此 trace 记录 `max_delta/avg_delta` 更贴近原逻辑。

## 7. 验证方式

在 Linux 主机上编译：

```bash
cmake --build build-orth --target MODULE_HSOLVER_ppcg MODULE_HSOLVER_bpcg MODULE_HSOLVER_cg MODULE_HSOLVER_dav -j$(nproc)
```

跑原测试，确保默认关闭 trace 时不影响行为：

```bash
ctest --test-dir build-orth -R "MODULE_HSOLVER_(ppcg|bpcg|cg|dav)$" --output-on-failure
```

开启 trace 后运行一个测试或 benchmark：

```bash
export ABACUS_DIAGO_TRACE=1
export ABACUS_DIAGO_TRACE_FILE=diago_trace.csv
ctest --test-dir build-orth -R "MODULE_HSOLVER_ppcg$" --output-on-failure
head diago_trace.csv
```

预期 CSV 头：

```text
solver,iter,nband,max_residual,avg_residual,n_converged,orth_error,note
```

## 8. 常见坑

- 不要在默认情况下打开 trace，否则 benchmark 会被文件 I/O 干扰。
- 不要每轮无条件从 GPU 拷 residual 到 host；必须放在 `trace.enabled()` 内部。
- Davidson 记录的是 eigenvalue delta，不要硬叫 residual。
- 如果新分支已经重构了函数名，按“计算 residual 后、收敛判断前后”的语义找插入点。
- 如果多个 MPI rank 同时写同一个 CSV，可能会有并发写入问题。当前设计主要用于单 rank 或测试场景；如果要用于大规模 MPI benchmark，建议每个 rank 写不同文件名，或只在指定 rank 写。
