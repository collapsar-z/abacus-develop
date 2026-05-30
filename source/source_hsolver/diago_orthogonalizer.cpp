#include "source_hsolver/diago_orthogonalizer.h"

#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"

#include <ATen/kernels/lapack.h>

#include <algorithm>
#include <cmath>
#include <complex>

namespace hsolver
{

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::set_dimensions(const int nbasis, const int ndim, const int nwork)
{
    this->n_basis = nbasis;
    this->n_dim = ndim;
    this->n_work = nwork;
}

template <typename T, typename Device>
T DiagoOrthogonalizer<T, Device>::inner_product(const T* lhs, const T* rhs) const
{
    T result = T(0);
    for (int ig = 0; ig < this->n_dim; ++ig)
    {
        result += std::conj(lhs[ig]) * rhs[ig];
    }
    Parallel_Reduce::reduce_pool(&result, 1);
    return result;
}

template <typename T, typename Device>
typename DiagoOrthogonalizer<T, Device>::Real DiagoOrthogonalizer<T, Device>::vector_norm(const T* vec) const
{
    const Real norm2 = std::max(Real(0), std::real(this->inner_product(vec, vec)));
    return std::sqrt(norm2);
}

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::scale_vector(T* vec, const Real alpha) const
{
    for (int ig = 0; ig < this->n_dim; ++ig)
    {
        vec[ig] *= alpha;
    }
    for (int ig = this->n_dim; ig < this->n_basis; ++ig)
    {
        vec[ig] = T(0);
    }
}

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::axpy_vector(T* y, const T* x, const T alpha) const
{
    for (int ig = 0; ig < this->n_dim; ++ig)
    {
        y[ig] += alpha * x[ig];
    }
}

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::modified_gram_schmidt(T* psi_in, std::vector<T>& hpsi_in) const
{
    for (int ib = 0; ib < this->n_work; ++ib)
    {
        T* xi = psi_in + ib * this->n_basis;
        T* hxi = hpsi_in.data() + ib * this->n_basis;
        for (int jb = 0; jb < ib; ++jb)
        {
            const T* xj = psi_in + jb * this->n_basis;
            const T* hxj = hpsi_in.data() + jb * this->n_basis;
            const T coeff = this->inner_product(xj, xi);
            this->axpy_vector(xi, xj, -coeff);
            this->axpy_vector(hxi, hxj, -coeff);
        }

        const Real norm = this->vector_norm(xi);
        if (norm <= Real(1.0e-14))
        {
            ModuleBase::WARNING_QUIT("DiagoOrthogonalizer::modified_gram_schmidt",
                                     "linear dependent wavefunctions");
        }
        this->scale_vector(xi, Real(1) / norm);
        this->scale_vector(hxi, Real(1) / norm);
    }
}

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::orth_cholesky(T* psi_in,
                                                   std::vector<T>& hpsi_in,
                                                   std::vector<T>& workspace) const
{
    std::vector<T> s(this->n_work * this->n_work, T(0));
    for (int col = 0; col < this->n_work; ++col)
    {
        for (int row = 0; row < this->n_work; ++row)
        {
            s[row + col * this->n_work]
                = this->inner_product(psi_in + row * this->n_basis, psi_in + col * this->n_basis);
        }
    }

    ct::kernels::lapack_potrf<T, ct::DEVICE_CPU>()('U', this->n_work, s.data(), this->n_work);

    for (int col = 0; col < this->n_work; ++col)
    {
        for (int row = col + 1; row < this->n_work; ++row)
        {
            s[row + col * this->n_work] = T(0);
        }
    }

    ct::kernels::lapack_trtri<T, ct::DEVICE_CPU>()('U', 'N', this->n_work, s.data(), this->n_work);

    this->rotate_block(psi_in, s, workspace);
    this->rotate_block(hpsi_in.data(), s, workspace);
}

template <typename T, typename Device>
bool DiagoOrthogonalizer<T, Device>::check_orthonormality(T* psi_in, const Real tolerance) const
{
    Real frob2 = 0;
    for (int col = 0; col < this->n_work; ++col)
    {
        for (int row = 0; row < this->n_work; ++row)
        {
            const T s = this->inner_product(psi_in + row * this->n_basis, psi_in + col * this->n_basis);
            const T delta = s - static_cast<T>(row == col ? 1.0 : 0.0);
            frob2 += std::norm(delta);
        }
    }
    return std::sqrt(frob2) < tolerance;
}

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::rotate_block(T* block,
                                                  const std::vector<T>& coeff,
                                                  std::vector<T>& workspace) const
{
    std::fill(workspace.begin(), workspace.end(), T(0));
    for (int out = 0; out < this->n_work; ++out)
    {
        T* dst = workspace.data() + out * this->n_basis;
        for (int in = 0; in < this->n_work; ++in)
        {
            const T* src = block + in * this->n_basis;
            const T c = coeff[in + out * this->n_work];
            for (int ig = 0; ig < this->n_dim; ++ig)
            {
                dst[ig] += src[ig] * c;
            }
        }
    }
    std::copy(workspace.begin(), workspace.end(), block);
}

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::rayleigh_ritz(T* psi_in,
                                                   std::vector<T>& hpsi_in,
                                                   std::vector<Real>& eigen,
                                                   std::vector<T>& workspace) const
{
    if (this->n_work == 0)
    {
        return;
    }

    std::vector<T> hsub(this->n_work * this->n_work, T(0));
    for (int col = 0; col < this->n_work; ++col)
    {
        for (int row = 0; row < this->n_work; ++row)
        {
            hsub[row + col * this->n_work]
                = this->inner_product(psi_in + row * this->n_basis, hpsi_in.data() + col * this->n_basis);
        }
    }

    ct::kernels::lapack_heevd<T, ct::DEVICE_CPU>()(this->n_work, hsub.data(), this->n_work, eigen.data());
    this->rotate_block(psi_in, hsub, workspace);
    this->rotate_block(hpsi_in.data(), hsub, workspace);
}

template <typename T, typename Device>
void DiagoOrthogonalizer<T, Device>::project_to_orthogonal_complement(T* psi_in, std::vector<T>& block) const
{
    for (int ib = 0; ib < this->n_work; ++ib)
    {
        T* vi = block.data() + ib * this->n_basis;
        for (int jb = 0; jb < this->n_work; ++jb)
        {
            const T* xj = psi_in + jb * this->n_basis;
            const T coeff = this->inner_product(xj, vi);
            this->axpy_vector(vi, xj, -coeff);
        }
    }
}

template class DiagoOrthogonalizer<std::complex<float>, base_device::DEVICE_CPU>;
template class DiagoOrthogonalizer<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class DiagoOrthogonalizer<std::complex<float>, base_device::DEVICE_GPU>;
template class DiagoOrthogonalizer<std::complex<double>, base_device::DEVICE_GPU>;
#endif

} // namespace hsolver
