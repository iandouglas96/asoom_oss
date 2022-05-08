#pragma once

#include <Eigen/Dense>

/*! 
 * Convert lat/long in degrees to UTM
 * See https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system#Simplified_formulae
 *
 * @param gps_latlong Point in (lat, long) in degrees
 * @param zone Set equal to UTM zone number, negative for southern version
 */
inline Eigen::Vector2d LatLong2UTM(const Eigen::Vector2d& gps_latlong, int& zone) {
  constexpr double n = 0.0016792203863837047; // f/(2-f)
  constexpr double A = 6367449.145823415;
  const Eigen::Vector3d alpha(
      0.0008377318188192541,
      7.608496958699166e-07,
      1.2034877875966646e-09
      );
  constexpr double k0 = 0.9996;
  constexpr double E0 = 500e3;
  double N0 = 0;

  // Figure out the zone
  // longitude is positive east, negative west
  zone = std::floor((gps_latlong[1] + 180)/6) + 1;
  // Goes down the middle of each zone
  double lambda0 = (static_cast<double>(zone)-0.5)*6 - 180;
  // Convert to radians
  lambda0 = lambda0 * M_PI / 180;
  Eigen::Vector2d latlong = gps_latlong * M_PI / 180;
  // Notated phi and lambda on wikipedia, so makes translating easier
  double phi = latlong[0];
  double lambda = latlong[1];

  if (gps_latlong[0] < 0) {
    // Southern hemisphere
    N0 = 10000e3;
    zone *= -1;
  }

  Eigen::Vector2d utm;

  // Intermediate values
  double tmp = 2*sqrt(n)/(1+n);
  double t = sinh(atanh(sin(phi)) - tmp * atanh(tmp * sin(phi)));
  double xi = atan(t/cos(lambda - lambda0));
  double eta = atanh(sin(lambda - lambda0)/sqrt(1 + t*t));

  // calculate sums
  double esum = 0;
  double nsum = 0;
  for (int j=1; j<=3; j++) {
    esum += alpha[j-1] * cos(2*j*xi) * sinh(2*j*eta);
    nsum += alpha[j-1] * sin(2*j*xi) * cosh(2*j*eta);
  }

  // final calcs
  utm[0] = E0 + k0*A*(eta + esum);
  utm[1] = N0 + k0*A*(xi + nsum);

  return utm;
}
