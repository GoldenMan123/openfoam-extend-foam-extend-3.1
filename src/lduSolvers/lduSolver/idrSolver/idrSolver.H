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

Class
    idrSolver

Description
    Preconditioned Induced Dimension Reduction solver with run-time selectable preconditioning.

Author
    Vladimir Zhukov. All rights reserved.

SourceFiles
    idrSolver.C

\*---------------------------------------------------------------------------*/

#ifndef idrSolver_H
#define idrSolver_H

#include "lduMatrix.H"
#include "Field.H"
#include "SubField.H"
#include "FieldFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

typedef FieldField<Field, scalar> scalarFieldField;

/*---------------------------------------------------------------------------*\
                         Class idrSolver Declaration
\*---------------------------------------------------------------------------*/

class idrSolver
:
    public lduSolver
{

    // Private data

        //- Preconditioner
        autoPtr<lduPreconditioner> preconPtr_;
    
        //- Run-time parameters
        label s_;
        scalar angle;

    // Private Member Functions

        //- Disallow default bitwise copy construct
        idrSolver(const idrSolver&);
    
        //- Disallow default bitwise assignment
        void operator=(const idrSolver&);

        void mul(scalarField &res, const scalarFieldField &a, const scalarField &b) const;
        void mul(scalarField &res, const scalarFieldField &a, const scalarField &b, label x1, label x2,
            label y1, label y2) const;
        void tmul(scalarField &res, const scalarFieldField &a, const scalarField &b, label x1, label x2,
            label y1, label y2) const;
        void relax(scalarField &dst, const scalarField &src, scalar coeff) const;
        void gauss(scalarField &c, const scalarFieldField &a, const scalarField &b, label x1, label x2) const;
        scalarField subvector(const scalarField &x, label s, label f) const;
        void generate(scalarFieldField &mtx, const scalarField &r, const scalarField &seed) const;


public:

    //- Runtime type information
    TypeName("IDR");

    // Constructors

        //- Construct from matrix components and solver data stream
        idrSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& coupleBouCoeffs,
            const FieldField<Field, scalar>& coupleIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary &dict
        );

    // Destructor

        virtual ~idrSolver()
        {}

    // Member Functions

        //- Solve the matrix with this solver
        virtual lduSolverPerformance solve
        (
            scalarField& x,
            const scalarField& b,
            const direction cmpt = 0
        ) const;

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

/// ************************************************************************* //
