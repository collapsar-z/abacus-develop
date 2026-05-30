#ifndef DIAGO_ORTHOGONALIZER_H_
#define DIAGO_ORTHOGONALIZER_H_

#include "source_base/macros.h"
#include "source_base/module_device/types.h"

#include <vector>

namespace hsolver
{

/**
 * @brief Shared orthogonalization helper for iterative eigensolvers.
 *
 * The helper owns no wavefunction memory. It operates on column-major blocks
 * with leading dimension n_basis and only uses the first n_dim entries of each
 * column in inner products and vector updates.
 */
template <typename T, typename Device = base_device::DEVICE_CPU>
class DiagoOrthogonalizer
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    DiagoOrthogonalizer() = default;

    DiagoOrthogonalizer(const int nbasis, const int ndim, const int nwork)
    {
        this->set_dimensions(nbasis, ndim, nwork);
    }

    void set_dimensions(const int nbasis, const int ndim, const int nwork);

    void modified_gram_schmidt(T* psi_in, std::vector<T>& hpsi_in) const;

    void orth_cholesky(T* psi_in, std::vector<T>& hpsi_in, std::vector<T>& workspace) const;

    bool check_orthonormality(T* psi_in, const Real tolerance = Real(1e-1)) const;

    void rotate_block(T* block, const std::vector<T>& coeff, std::vector<T>& workspace) const;

    void rayleigh_ritz(T* psi_in,
                       std::vector<T>& hpsi_in,
                       std::vector<Real>& eigen,
                       std::vector<T>& workspace) const;

    void project_to_orthogonal_complement(T* psi_in, std::vector<T>& block) const;

  private:
    T inner_product(const T* lhs, const T* rhs) const;
    Real vector_norm(const T* vec) const;
    void scale_vector(T* vec, const Real alpha) const;
    void axpy_vector(T* y, const T* x, const T alpha) const;

    int n_basis = 0;
    int n_dim = 0;
    int n_work = 0;
};

} // namespace hsolver

#endif
