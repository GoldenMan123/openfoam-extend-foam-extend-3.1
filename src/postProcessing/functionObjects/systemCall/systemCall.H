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
    Foam::systemCall

Description
    Executes system calls, entered in the form of a string list

SourceFiles
    systemCall.C
    IOsystemCall.H

\*---------------------------------------------------------------------------*/

#ifndef systemCall_H
#define systemCall_H

#include "stringList.H"
#include "pointFieldFwd.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of classes
class objectRegistry;
class dictionary;
class mapPolyMesh;

/*---------------------------------------------------------------------------*\
                       Class systemCall Declaration
\*---------------------------------------------------------------------------*/

class systemCall
{
protected:

    // Private data

        //- Name of this set of system calls
        word name_;

        //- List of calls to execute - every step
        stringList executeCalls_;

        //- List of calls to execute when exiting the time-loop
        stringList endCalls_;

        //- List of calls to execute - write steps
        stringList writeCalls_;


    // Private Member Functions

        //- Disallow default bitwise copy construct
        systemCall(const systemCall&);

        //- Disallow default bitwise assignment
        void operator=(const systemCall&);


public:

    //- Runtime type information
    TypeName("systemCall");


    // Constructors

        //- Construct for given objectRegistry and dictionary.
        //  Allow the possibility to load fields from files
        systemCall
        (
            const word& name,
            const objectRegistry& unused,
            const dictionary&,
            const bool loadFromFilesUnused = false
        );


    // Destructor

        virtual ~systemCall();


    // Member Functions

        //- Return name of the system call set
        virtual const word& name() const
        {
            return name_;
        }

        //- Read the system calls
        virtual void read(const dictionary&);

        //- Execute the "executeCalls" at each time-step
        virtual void execute();

        //- Execute the "endCalls" at the final time-loop
        virtual void end();

        //- Write, execute the "writeCalls"
        virtual void write();

        //- Update for changes of mesh
        virtual void updateMesh(const mapPolyMesh&)
        {}

        //- Update for changes of mesh
        virtual void movePoints(const pointField&)
        {}
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
