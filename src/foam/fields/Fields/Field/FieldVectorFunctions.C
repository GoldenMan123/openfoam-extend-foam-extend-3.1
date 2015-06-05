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

\*---------------------------------------------------------------------------*/

#include "FieldVectorFunctions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/* * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * */

#define VECTOR_SIZE 2

inline
scalar vectorSumProdImpl(const scalar *a, const scalar *b, label size)
{
    typedef scalar vst __attribute__((vector_size(VECTOR_SIZE * sizeof(scalar))));
    vst tmp = {0};
    const vst * __restrict__ a_ = reinterpret_cast<const vst *>(a);
    const vst * __restrict__ b_ = reinterpret_cast<const vst *>(b);
    for (label i = 0; i < size / VECTOR_SIZE; ++i) {
        tmp += a_[i] * b_[i];
    }
    scalar res = 0.0;
    for (label i = 0; i < VECTOR_SIZE; ++i) {
        res += tmp[i];
    }
    return res;
}

scalar vectorSumProd(const scalar *a, const scalar *b, label size)
{
    unsigned long a_ = reinterpret_cast<unsigned long>(a) & (VECTOR_SIZE * sizeof(scalar) - 1) / sizeof(scalar);
    unsigned long b_ = reinterpret_cast<unsigned long>(b) & (VECTOR_SIZE * sizeof(scalar) - 1) / sizeof(scalar);
    scalar tmp = 0;
    if (a_ == b_ && size >= 4 * VECTOR_SIZE) {
        label sh = static_cast<label>(VECTOR_SIZE - a_);
        if (a_) {
            for (label i = 0; i < sh; ++i) {
                tmp += a[i] * b[i];
            }
            size -= sh;
        } else {
            sh = 0;
        }
        for (label i = size & ~(VECTOR_SIZE - 1); i < size; ++i) {
            tmp += (a + sh)[i] * (b + sh)[i];
        }
        tmp += vectorSumProdImpl(a + sh, b + sh, size & ~(VECTOR_SIZE - 1));
        return tmp;
    }
    for (label i = 0; i < size; ++i) {
        tmp += a[i] * b[i];
    }
    return tmp;
}

template<>
scalar sumProd<scalar>(const UList<scalar>& f1, const UList<scalar>& f2)
{
    if (f1.size() && (f1.size() == f2.size()))
    {
        return vectorSumProd(&f1[0], &f2[0], f1.size());
    }
    else
    {
        return 0.0;
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// ************************************************************************* //
