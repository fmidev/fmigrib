#include <boost/assign.hpp>
#include <boost/bimap.hpp>

typedef boost::bimap<long, long> bimapLong;

// GRIB1 <--> GRIB2

bimapLong gridTypeMap = boost::assign::list_of<bimapLong::relation>(0, 0)  // ll
    (10, 1)                                                                // rll
    (20, 2)                                                                // stretched ll
    (30, 3)                                                                // stretched rll
    (5, 20)                                                                // polar stereographic
    (3, 30)                                                                // lambert conformal
    (4, 40)                                                                // gaussian ll
    ;

bimapLong levelTypeMap = boost::assign::list_of<bimapLong::relation>(1, 1)  // ground
    (8, 8)                                                                  // top of atmosphere
    (6, 6)                                                                  // max wind level
    (100, 100)                                                              // isobaric
    (160, 160)                                                              // depth below sea
    (102, 101)                                                              // mean sea
    (103, 102)                                                              // specific altitude above mean-sea level
    (105, 103)                                                              // specified height above ground
    (101, 108)                                                              // pressure deviation from ground to level
    (109, 105)                                                              // hybrid
    (111, 106)                                                              // depth below land surface
    (200, 10)   // entire atmosphere considered as a single layer
    (246, 246)  // max thetae
    (7, 7)      // tropopause
    ;
