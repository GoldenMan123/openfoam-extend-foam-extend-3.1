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
    BlockIDRSolver
    
Description
    Preconditioned Induced Dimension Reduction solver.

Author
    Vladimir Zhukov. All rights reserved.

SourceFiles
    BlockIDRSolver.C

\*---------------------------------------------------------------------------*/

#ifndef BlockIDRSolver_H
#define BlockIDRSolver_H

#include "blockLduSolvers.H"
#include "BlockIterativeSolver.H"
#include "blockLduPrecons.H"
#include "SubField.H"
#include "Random.H"
#include "globalIndex.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

typedef FieldField<Field, scalar> scalarFieldField;

/*---------------------------------------------------------------------------*\
                     Class BlockIDRSolver Declaration
\*---------------------------------------------------------------------------*/

template<class Type>
class BlockIDRSolver
:
    public BlockIterativeSolver<Type>
{

    // Private data

        //- Preconditioner
        autoPtr<BlockLduPrecon<Type> > preconPtr_;

        //- Run-time parameters
        label s_;
        scalar angle;
   
    // Private Member Functions

        //- Disallow default bitwise copy construct
        BlockIDRSolver(const BlockIDRSolver<Type>&);
   
        //- Disallow default bitwise assignment
        void operator=(const BlockIDRSolver<Type>&);

        void mul(scalarField &res, const scalarFieldField &a, const scalarField &b) const;
        void mul(scalarField &res, const scalarFieldField &a, const scalarField &b, label x1, label x2,
            label y1, label y2) const;
        void tmul(scalarField &res, const scalarFieldField &a, const scalarField &b, label x1, label x2,
            label y1, label y2) const;
        void relax(scalarField &dst, const scalarField &src, scalar coeff) const;
        void gauss(scalarField &c, const scalarFieldField &a, const scalarField &b, label x1, label x2) const;
        scalarField subvector(const scalarField &x, label s, label f) const;
        void generate(scalarFieldField &mtx, const scalarField &r, const Field<label> &seed) const;
        
        //- Convertors
        template<class Type2> void convert_in(scalarField &dst, const Field<Type2> &src) const;
        void convert_in(scalarField &dst, const Field<scalar> &src) const;
        void convert_in(scalarField &dst, const Field<tensor> &src) const;
        template<class Type2> void convert_out(Field<Type2> &dst, const scalarField &src) const;
        void convert_out(Field<scalar> &dst, const scalarField &src) const;
        void convert_out(Field<tensor> &dst, const scalarField &src) const;
        template<class Type2> void alloc_in(scalarField &dst, const Field<Type2> &src) const;
        void alloc_in(scalarField &dst, const Field<scalar> &src) const;
        void alloc_in(scalarField &dst, const Field<tensor> &src) const;

public:

    //- Runtime type information
    TypeName("IDR");

    // Constructors

        //- Construct from matrix components and solver data stream
        BlockIDRSolver
        (
            const word& fieldName,
            const BlockLduMatrix<Type>& matrix,
            const dictionary& dict
        );

    // Destructor
        
        virtual ~BlockIDRSolver()
        {}
            
    // Member Functions

        //- Solve the matrix with this solver
        virtual BlockSolverPerformance<Type> solve
        (
            Field<Type>& x,
            const Field<Type>& b
        );
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#ifdef NoRepository
#   include "BlockIDRSolver.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
