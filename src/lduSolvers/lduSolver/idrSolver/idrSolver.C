/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | foam-extend: Open Source CFD
   \\    /   O peration     |
    \\  /    A nd           | For copyright notice see file Copyright
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of foam-extend.

    foam-extend is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    foam-extend is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with foam-extend.  If not, see <http://www.gnu.org/licenses/>.

Description
    Preconditioned Induced Dimension Reduction solver with run-time selectable preconditioning

Author
    Vladimir Zhukov. All rights reserved.

\*---------------------------------------------------------------------------*/

#include "idrSolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(idrSolver, 0);

    lduSolver::addsymMatrixConstructorToTable<idrSolver>
        addidrSolverSymMatrixConstructorToTable_;

    lduSolver::addasymMatrixConstructorToTable<idrSolver>
        addidrSolverAsymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

//- Construct from matrix and solver data stream
Foam::idrSolver::idrSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& coupleBouCoeffs,
    const FieldField<Field, scalar>& coupleIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& dict
)
:
    lduSolver
    (
        fieldName,
        matrix,
        coupleBouCoeffs,
        coupleIntCoeffs,
        interfaces,
        dict
    ),
    preconPtr_
    (
        lduPreconditioner::New
        (
            matrix,
            coupleBouCoeffs,
            coupleIntCoeffs,
            interfaces,
            dict
        )
    ),
    s_(readLabel(dict.lookup("s"))),
    angle(readScalar(dict.lookup("angle")))
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::idrSolver::mul(scalarField &res, const scalarFieldField &a, const scalarField &b) const
{
    for (label i = 0; i < a.size(); ++i) {
        res[i] = sumProd(a[i], b);
    }
}

void Foam::idrSolver::mul(scalarField &res, const scalarFieldField &a, const scalarField &b,
    label x1, label x2, label y1, label y2) const
{
    for (label i = 0; i < x2 - x1 + 1; ++i) { 
        res[i] = sumProd(SubField<scalar>(a[i + x1], y2 - y1  + 1, y1), SubField<scalar>(b, y2 - y1 + 1, y1));
    }
}

void Foam::idrSolver::tmul(scalarField &res, const scalarFieldField &a, const scalarField &b,
    label x1, label x2, label y1, label y2) const
{
    label nI = x2 - x1 + 1;
    const scalar * __restrict__ bR = &b[0];
    scalar * __restrict__ resI_ = &res[0];
    scalar * __restrict__ resI = static_cast<scalar *>(__builtin_assume_aligned(resI_, 16));
    label y1I = y1;
    label x1I = x1;

    static const label L1_CACHED_PART = 16 * 1024 / sizeof(scalar);
    label parts = nI / L1_CACHED_PART + ((nI % L1_CACHED_PART) ? 1 : 0);

    for (label z = 0; z < parts; ++z) {
        label part_size = L1_CACHED_PART;
        if ((z == parts - 1) && (nI % part_size)) {
            part_size = nI % part_size;
        }
        {
            scalar bI = bR[0];
            const scalar * __restrict__ aI_ = &a[y1I][0];
            const scalar * __restrict__ aI = static_cast<scalar *>(__builtin_assume_aligned(aI_, 16));
            for (label i = 0; i < part_size; ++i) {
                resI[i + z * L1_CACHED_PART] = aI[i + x1I + z * L1_CACHED_PART] * bI;
            }
        }
        for (label j = 1; j < y2 - y1 + 1; ++j) {
            scalar bI = bR[j];
            const scalar * __restrict__ aI_ = &a[j + y1I][0];
            const scalar * __restrict__ aI = static_cast<scalar *>(__builtin_assume_aligned(aI_, 16));
            for (label i = 0; i < part_size; ++i) {
                resI[i + z * L1_CACHED_PART] += aI[i + x1I + z * L1_CACHED_PART] * bI;
            }
        }
    }
}

void Foam::idrSolver::gauss(scalarField &c, const scalarFieldField &a, const scalarField &b,
    label x1, label x2) const
{
    label n = x2 - x1 + 1;
    for (label i = 0; i < n; ++i) {
        c[i] = b[i + x1];
        for (label j = 0; j < i; ++j) {
            c[i] -= a[i + x1][j + x1] * c[j];
        }
        c[i] /= a[i + x1][i + x1];
    }
}

Foam::scalarField Foam::idrSolver::subvector(const scalarField &x, label s, label f) const
{
    scalarField res(f - s + 1);
    for (label i = s; i <= f; ++i) {
        res[i - s] = x[i];
    }
    return res;
}

void Foam::idrSolver::relax(scalarField &dst, const scalarField &src, scalar coeff) const 
{
    scalar c = coeff;
    label nI = src.size();
    scalar * __restrict__ dstI_ = &dst[0];
    scalar * __restrict__ dstI = static_cast<scalar *>(__builtin_assume_aligned(dstI_, 16));
    const scalar * __restrict__ srcI_ = &src[0];
    const scalar * __restrict__ srcI = static_cast<scalar *>(__builtin_assume_aligned(srcI_, 16));
    for (label i = 0; i < nI; ++i) {
        dstI[i] -= srcI[i] * c;
    }
}

void Foam::idrSolver::generate(scalarFieldField &mtx, const scalarField &r, const scalarField &seed) const
{
    label n = mtx.size();
    label m = mtx[0].size();
    mtx[0] = r;
    for (label j = 0; j < m; ++j) {
        const unsigned long long *seed_ptr = reinterpret_cast<const unsigned long long *>(&seed[j]);
        Random rng(*seed_ptr);
        for (label i = 1; i < n; ++i) {
            mtx[i][j] = (2.0 * rng.scalar01() - 1.0);
        }
    }
    for (label i = 1; i < n; ++i) {
        double scal;
        for (label k = 0; k < i; ++k) {
            scal = gSumProd(mtx[i], mtx[k]) / gSumProd(mtx[k], mtx[k]);
            mtx[i] -= scal * mtx[k];
        }
    }
}

Foam::lduSolverPerformance Foam::idrSolver::solve
(
    scalarField& x,
    const scalarField& b,
    const direction cmpt
) const
{
    lduSolverPerformance solverPerf(typeName, fieldName());

    scalarField wA(x.size());
    scalarField rA(x.size());

    // Compute initial residual:
    matrix_.Amul(wA, x, coupleBouCoeffs_, interfaces_, cmpt);

    scalar normFactor = this->normFactor(x, b, wA, rA, cmpt);

    forAll(rA, i)
    {
        rA[i] = b[i] - wA[i];
    }

    solverPerf.initialResidual() = gSumMag(rA) / normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if (!stop(solverPerf))
    {

        scalarFieldField P(s_);
        forAll(P, i) {
            P.set(i, new scalarField(x.size()));
        }

        scalarField seed(b.size());
        Random rng(877117);
        for (label i = 0; i < Pstream::myProcNo() + 1; ++i) {
            forAll(b, i) {
                seed[i] = rng.scalar01();
            }
        }
        
        generate(P, rA, seed);

        // Initialize G(n, s) = 0, U(n, s) = 0 and M(s, s) = I

        scalarFieldField G(s_), U(s_), M(s_);
        forAll(G, i) {
            G.set(i, new scalarField(x.size()));
        }
        forAll(U, i) {
            U.set(i, new scalarField(x.size()));
        }
        forAll(M, i) {
            M.set(i, new scalarField(s_));
        }

        for (label i = 0; i < s_; ++i) {
            for (label j = 0; j < x.size(); ++j) {
                G[i][j] = 0;
            }
        }

        for (label i = 0; i < s_; ++i) {
            for (label j = 0; j < x.size(); ++j) {
                U[i][j] = 0;
            }
        }

        for (label i = 0; i < s_; ++i) {
            for (label j = 0; j < s_; ++j) {
                M[i][j] = ((i == j) ? 1 : 0);
            }
        }

        // Initialize omega = 1

        scalar om = 1;

        // Main interation loop

        scalarField v(x.size());
        scalarField c(s_);
        scalarField t(x.size());
        scalarField f(s_);
        scalarField tmp3(s_);
        scalarField tmp1(x.size());

        label nIters = 0;

        do
        {
            // New right-hand for small system

            mul(f, P, rA); 
            reduce(f, sumOp<scalarField>()); //for MPI

            ++nIters;

            for (label k = 0; k < s_; ++k) {

                // Solve small system and make v orthogonal to P

                if (solverPerf.checkSingularity(sumMag(subvector(f, k, s_ - 1)))) {
                    forAll(c, i) {
                        c[i] = 0;
                    }
                } else {
                    gauss(c, M, f, k, s_  - 1); 
                }

                // Preconditioning
                tmul(tmp1, G, c, 0, x.size() - 1, k, s_ - 1);
                preconPtr_->precondition(v, rA - tmp1, cmpt);

                // Compute new U and G

                tmul(tmp1, U, c, 0, x.size() - 1, k, s_ - 1);
                relax(tmp1, v, -om);
                U[k] = tmp1;

                matrix_.Amul(G[k], U[k], coupleBouCoeffs_, interfaces_, cmpt);

                // Bi-Orthogonalize the nw basis vectors

                for (label i = 0; i < k; ++i) {
                    scalar alpha = gSumProd(P[i], G[k])  / M[i][i];
                    relax(G[k], G[i], alpha);
                    relax(U[k], U[i], alpha);
                }

                // New column of M = P.T() * G (first k - 1 entries are zero)

                mul(tmp3, P, G[k], k, s_ - 1, 0, x.size() - 1); 
                reduce(tmp3, sumOp<scalarField>()); //for MPI

                for (label i = 0; i < s_ - k; ++i) {
                    M[i + k][k] = tmp3[i];
                }

                // Make rA orthoronal to q_i, i = 1..k

                if (solverPerf.checkSingularity(mag(M[k][k]))) {
                    Info << "Singularity! M[k][k] = " << M[k][k] << endl;
                    break;
                }

                scalar beta = f[k] / M[k][k];

                relax(rA, G[k], beta);
                relax(x, U[k], -beta);

                // Calculate new f = P.T() * rA (first k components are zero)

                if (k < s_ - 1) {
                    for (label i = k + 1; i < s_; ++i) {
                        f[i] -= beta * M[i][k];
                    }
                }

            }

            // Preconditioning

            preconPtr_->precondition(v, rA, cmpt);

            matrix_.Amul(t, v, coupleBouCoeffs_, interfaces_, cmpt);

            // Calculate omega

            scalar ra_norm = sqrt(gSumProd(rA, rA));
            scalar t_norm = sqrt(gSumProd(t, t));
            scalar ts = gSumProd(t, rA);
            scalar rho = mag(ts / (ra_norm * t_norm));
            om = ts / (gSumProd(t, t));
            if (rho < angle) {
                om = om * angle / rho;
            }

            if (solverPerf.checkSingularity(mag(om))) {
                Info << "Singularity! om = " << om << endl;
                break;
            }
            // Update rA and x

            relax(rA, t, om);
            relax(x, v, -om); 

            // Smoothing

            solverPerf.finalResidual() = gSumMag(rA) / normFactor;
            solverPerf.nIterations()++;

        } while (!stop(solverPerf));

    }

    return solverPerf;
}


// ************************************************************************* //
