# 第六题报告：特征值求解器代码修改与重构

## 1. 本周工作主题

本周的主题是代码的修改与重构。我的工作重点不是重新实现一个新的特征值求解算法，而是阅读现有 `source_hsolver` 中的代码，找出其中可以整理的公共逻辑，并做一次范围可控的模块化重构。

本次选择的重构对象是 PPCG 求解器中的正交化相关代码。原因是这部分代码比较集中，而且正交化、投影、Rayleigh-Ritz 子空间对角化等操作在多个特征值求解器中都会出现，适合作为代码重构的切入点。

涉及的主要文件包括：

- `diago_ppcg.cpp`
- `diago_ppcg.h`
- `diago_orthogonalizer.cpp`
- `diago_orthogonalizer.h`
- `test/diago_orthogonalizer_test.cpp`
- `CMakeLists.txt`
- `test/CMakeLists.txt`

## 2. 原代码中的问题

在原来的 `DiagoPPCG` 中，PPCG 主迭代流程和一些通用线性代数操作写在同一个类里。例如：

- `modified_gram_schmidt`
- `orth_cholesky`
- `check_orthonormality`
- `rotate_block`
- `rayleigh_ritz`
- `project_to_orthogonal_complement`

这些函数虽然被 PPCG 使用，但它们本身并不是 PPCG 独有的算法逻辑。它们主要完成的是正交化、投影、子空间矩阵构造和向量旋转等通用操作。

这样写有几个问题：

1. `DiagoPPCG` 类职责比较多，主算法流程和底层辅助操作混在一起。
2. 正交化相关函数以后不方便被其他求解器复用。
3. 如果以后想优化正交化部分，例如改成 GEMM 或减少 MPI 归约次数，需要在 solver 类内部改动，代码边界不够清楚。

因此，本次重构的目标是把 PPCG 中这些通用的正交化操作抽出来，形成一个独立的辅助模块。

## 3. 重构思路

我新增了一个辅助类：

```cpp
DiagoOrthogonalizer<T, Device>
```

对应文件为：

```text
diago_orthogonalizer.h
diago_orthogonalizer.cpp
```

这个类主要负责以下操作：

- Modified Gram-Schmidt 正交化；
- Cholesky 正交化；
- 检查波函数是否正交归一；
- 对一组向量进行矩阵旋转；
- Rayleigh-Ritz 子空间对角化；
- 将向量投影到当前子空间的正交补上。

重构后，`DiagoPPCG` 主要保留 PPCG 自身的迭代逻辑，例如残差计算、收敛判断、band locking 和小子空间更新；而正交化相关的公共操作交给 `DiagoOrthogonalizer` 完成。

## 4. 具体代码修改

### 4.1 新增正交化模块

新增文件：

```text
diago_orthogonalizer.h
diago_orthogonalizer.cpp
```

主要接口如下：

```cpp
void set_dimensions(int nbasis, int ndim, int nwork);
void modified_gram_schmidt(T* psi, std::vector<T>& hpsi) const;
void orth_cholesky(T* psi, std::vector<T>& hpsi, std::vector<T>& workspace) const;
bool check_orthonormality(T* psi, Real tolerance = Real(1e-1)) const;
void rotate_block(T* block, const std::vector<T>& coeff, std::vector<T>& workspace) const;
void rayleigh_ritz(T* psi, std::vector<T>& hpsi, std::vector<Real>& eigen, std::vector<T>& workspace) const;
void project_to_orthogonal_complement(T* psi, std::vector<T>& block) const;
```

这个类不保存波函数数据，只保存维度信息。实际的 `psi`、`hpsi`、`work` 等数组仍然由 `DiagoPPCG` 管理。这样可以避免引入新的数据所有权问题。

### 4.2 修改 DiagoPPCG

在 `diago_ppcg.h` 中新增成员：

```cpp
DiagoOrthogonalizer<T, Device> orthogonalizer;
```

在 `DiagoPPCG::init_iter` 中初始化它的维度：

```cpp
this->orthogonalizer.set_dimensions(this->n_basis, this->n_dim, this->n_work);
```

然后将 `DiagoPPCG::diag` 中原来的正交化调用改为：

```cpp
this->orthogonalizer.modified_gram_schmidt(psi_in, this->hpsi);
this->orthogonalizer.rayleigh_ritz(psi_in, this->hpsi, this->eigen, this->work);
this->orthogonalizer.project_to_orthogonal_complement(psi_in, this->w);
this->orthogonalizer.orth_cholesky(psi_in, this->hpsi, this->work);
```

同时删除了 `DiagoPPCG` 类中对应的私有成员函数声明和实现，使 `DiagoPPCG` 的代码更集中于 PPCG 算法本身。

### 4.3 修改构建文件

在 `source_hsolver/CMakeLists.txt` 中加入：

```cmake
diago_orthogonalizer.cpp
```

在 `source_hsolver/test/CMakeLists.txt` 中，把新模块加入 PPCG 相关测试目标，并新增了一个单独测试：

```text
MODULE_HSOLVER_orthogonalizer
```

## 5. 单元测试设计

为了验证新抽出的模块是否正确，我新增了：

```text
test/diago_orthogonalizer_test.cpp
```

测试内容包括：

1. `modified_gram_schmidt` 后，检查 `X^H X` 是否接近单位矩阵。
2. `orth_cholesky` 后，检查 `X^H X` 是否接近单位矩阵。
3. `rotate_block` 是否符合列主序矩阵乘法的结果。
4. `project_to_orthogonal_complement` 后，检查目标向量在当前 `psi` 子空间上的分量是否被消去。

这些测试主要针对新模块本身，而不是完整的 PPCG 算法。这样可以更直接地验证本次重构抽取出的公共函数是否正确。

## 6. 验证方式

由于当前目录中没有现成的 `build` 构建目录，所以本次没有在本地实际运行完整编译和测试。后续如果在主仓库环境中验证，可以使用下面的命令：

```bash
cmake -B build -DBUILD_TESTING=ON -DENABLE_MPI=ON -DENABLE_LCAO=ON
cmake --build build --target MODULE_HSOLVER_orthogonalizer
ctest --test-dir build -V -R MODULE_HSOLVER_orthogonalizer
```

如果要进一步确认 PPCG 重构后没有破坏原有行为，可以继续运行：

```bash
cmake --build build --target MODULE_HSOLVER_ppcg
ctest --test-dir build -V -R MODULE_HSOLVER_ppcg
```

本次已做的静态检查是：

```bash
git diff --check
```

该检查没有发现补丁格式或空白字符问题。

## 7. 本次重构的效果

本次修改后，代码结构有以下变化：

1. `DiagoPPCG` 中减少了一部分正交化细节代码。
2. 正交化、投影、Rayleigh-Ritz 和向量旋转被集中到 `DiagoOrthogonalizer` 中。
3. 新增了针对公共模块的单元测试。
4. 后续如果要优化正交化部分，可以优先在 `DiagoOrthogonalizer` 中进行，而不需要直接修改 PPCG 主循环。

本次重构没有改变 PPCG 的算法思路，只是调整代码组织方式，使代码职责更清楚。

## 8. 没有继续整合其他求解器的原因

虽然 CG、Davidson、BPCG 中也存在类似的正交化逻辑，但本次没有一次性全部迁移，主要原因是：

1. BPCG 已经使用了 `PGemmCN` 和 `PLinearTransform` 等并行 GEMM 路径，和 PPCG 当前的 `std::vector<T>` 数据结构不完全一样。
2. CG 和 Davidson 的正交化流程与各自迭代逻辑结合较紧，直接迁移可能引入额外风险。
3. 本次作业的目标是完成一次清晰、可控的重构，而不是进行大范围重写。

因此，我采用了渐进式重构方式：先以 PPCG 为例完成模块抽取和单元测试，后续如果继续推进，可以再逐步考虑 CG、Davidson 和 BPCG 的统一接口。

## 9. 总结

本次作业围绕“代码修改与重构”展开，主要完成了 PPCG 正交化相关代码的模块化整理。通过新增 `DiagoOrthogonalizer`，将原本写在 `DiagoPPCG` 内部的通用正交化操作抽取出来，并补充了独立单元测试。

这次修改的重点不是提高性能，而是改善代码结构。重构后，PPCG 主流程更清楚，公共功能也有了独立模块，为后续进一步优化和复用提供了基础。
