Authors:
Charles Guse (Xylae)
Dana Vrajitoru (Shogoth)

Indiana University South Bend
Computer and Information Sciences Department
South Bend, IN 46545, USA

This application is based on the Windows version of the C++ client for the championship.

We've added the class EpicMount which extends the class WrapperBaseDriver that was provided
in the original package. This class also uses the class CarState.

Added files:
EpicMount.h
EpicMount.cpp

Modified files:
client.cpp
Modified to replace the generic class constant with the name of our class.

Compilation:
We've used Microsoft Visual Studio 2005, starting with a simple empty C++ project to which 
we added all of the files. No special requirements for the compilation and we think that
it should work the same with other IDEs.