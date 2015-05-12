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

Author
    Vladimir Zhukov. All rights reserved.

\*---------------------------------------------------------------------------*/

#include "idrPgen.H"

#include <ctime>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(idrPgen, 0);
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::idrPgen::idrPgen()
{
    srand(std::time(NULL));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::idrPgen::generate(scalarFieldField &mtx, const scalarField &r) const
{
    label n = mtx.size();
    label m = mtx[0].size();
    mtx[0] = r;
    for (label i = 1; i < n; ++i) {
        for (label j = 0; j < m; ++j) {
            mtx[i][j] = (2.0 * (1.0 + rand()) / (RAND_MAX + 1.0) - 1.0);
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


// ************************************************************************* //
