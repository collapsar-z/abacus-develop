#include "../diago_orthogonalizer.h"

#include "gtest/gtest.h"
#include "source_base/parallel_comm.h"

#include <cmath>
#include <complex>
#include <vector>

namespace
{

using T = std::complex<double>;
using Orthogonalizer = hsolver::DiagoOrthogonalizer<T, base_device::DEVICE_CPU>;

T dot(const std::vector<T>& lhs,
      const std::vector<T>& rhs,
      const int n_basis,
      const int n_dim,
      const int lhs_col,
      const int rhs_col)
{
    T result = T(0);
    const T* x = lhs.data() + lhs_col * n_basis;
    const T* y = rhs.data() + rhs_col * n_basis;
    for (int i = 0; i < n_dim; ++i)
    {
        result += std::conj(x[i]) * y[i];
    }
    return result;
}

void expect_orthonormal(const std::vector<T>& block, const int n_basis, const int n_dim, const int n_work)
{
    for (int col = 0; col < n_work; ++col)
    {
        for (int row = 0; row < n_work; ++row)
        {
            const T value = dot(block, block, n_basis, n_dim, row, col);
            const double expected = row == col ? 1.0 : 0.0;
            EXPECT_NEAR(value.real(), expected, 1.0e-10);
            EXPECT_NEAR(value.imag(), 0.0, 1.0e-10);
        }
    }
}

std::vector<T> make_independent_block()
{
    // Column-major block with n_basis = n_dim = 4 and n_work = 3.
    return {
        T(1.0, 0.0), T(2.0, 0.0), T(0.5, 0.5), T(-1.0, 0.0),
        T(0.0, 0.0), T(1.0, 0.0), T(1.0, -0.5), T(2.0, 0.0),
        T(1.0, 0.0), T(0.0, 1.0), T(2.0, 0.0), T(0.5, 0.0),
    };
}

} // namespace

TEST(DiagoOrthogonalizerTest, ModifiedGramSchmidtProducesOrthonormalBlock)
{
    const int n_basis = 4;
    const int n_dim = 4;
    const int n_work = 3;

    std::vector<T> psi = make_independent_block();
    std::vector<T> hpsi = psi;

    Orthogonalizer orth(n_basis, n_dim, n_work);
    orth.modified_gram_schmidt(psi.data(), hpsi);

    expect_orthonormal(psi, n_basis, n_dim, n_work);
    EXPECT_TRUE(orth.check_orthonormality(psi.data(), 1.0e-10));
}

TEST(DiagoOrthogonalizerTest, CholeskyProducesOrthonormalBlock)
{
    const int n_basis = 4;
    const int n_dim = 4;
    const int n_work = 3;

    std::vector<T> psi = make_independent_block();
    std::vector<T> hpsi = psi;
    std::vector<T> workspace(n_basis * n_work, T(0));

    Orthogonalizer orth(n_basis, n_dim, n_work);
    orth.orth_cholesky(psi.data(), hpsi, workspace);

    expect_orthonormal(psi, n_basis, n_dim, n_work);
    EXPECT_TRUE(orth.check_orthonormality(psi.data(), 1.0e-10));
}

TEST(DiagoOrthogonalizerTest, RotateBlockMatchesColumnMajorMatrixProduct)
{
    const int n_basis = 3;
    const int n_dim = 3;
    const int n_work = 2;

    std::vector<T> block = {
        T(1.0, 0.0), T(2.0, 0.0), T(3.0, 0.0),
        T(4.0, 0.0), T(5.0, 0.0), T(6.0, 0.0),
    };
    const std::vector<T> coeff = {
        T(2.0, 0.0), T(3.0, 0.0),
        T(-1.0, 0.0), T(0.5, 0.0),
    };
    std::vector<T> workspace(n_basis * n_work, T(0));

    Orthogonalizer orth(n_basis, n_dim, n_work);
    orth.rotate_block(block.data(), coeff, workspace);

    const std::vector<T> expected = {
        T(14.0, 0.0), T(19.0, 0.0), T(24.0, 0.0),
        T(1.0, 0.0), T(0.5, 0.0), T(0.0, 0.0),
    };
    ASSERT_EQ(block.size(), expected.size());
    for (size_t i = 0; i < block.size(); ++i)
    {
        EXPECT_NEAR(block[i].real(), expected[i].real(), 1.0e-12);
        EXPECT_NEAR(block[i].imag(), expected[i].imag(), 1.0e-12);
    }
}

TEST(DiagoOrthogonalizerTest, ProjectionRemovesComponentsInPsiSubspace)
{
    const int n_basis = 4;
    const int n_dim = 4;
    const int n_work = 2;

    std::vector<T> psi = {
        T(1.0, 0.0), T(0.0, 0.0), T(0.0, 0.0), T(0.0, 0.0),
        T(0.0, 0.0), T(1.0, 0.0), T(0.0, 0.0), T(0.0, 0.0),
    };
    std::vector<T> block = {
        T(2.0, 0.0), T(-1.0, 0.0), T(3.0, 0.0), T(4.0, 0.0),
        T(0.5, 0.0), T(7.0, 0.0), T(-2.0, 0.0), T(1.0, 0.0),
    };

    Orthogonalizer orth(n_basis, n_dim, n_work);
    orth.project_to_orthogonal_complement(psi.data(), block);

    for (int col = 0; col < n_work; ++col)
    {
        for (int row = 0; row < n_work; ++row)
        {
            const T value = dot(psi, block, n_basis, n_dim, row, col);
            EXPECT_NEAR(value.real(), 0.0, 1.0e-12);
            EXPECT_NEAR(value.imag(), 0.0, 1.0e-12);
        }
    }
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
    POOL_WORLD = MPI_COMM_WORLD;
#endif
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
