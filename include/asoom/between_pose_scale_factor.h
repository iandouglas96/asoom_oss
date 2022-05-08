// MODIFIED FROM BetweenFactor.h in GTSAM, GTSAM Copyright:
/* ----------------------------------------------------------------------------
 *
 *
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 *  @file  between_pose_scale_factor.h
 *  @author Frank Dellaert, Viorela Ila, Ian D. Miller
 **/
#pragma once

#include <ostream>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/Lie.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

  /**
   * A class for a measurement between two poses with constant scale factor
   * @addtogroup SLAM
   */
  class BetweenPoseScaleFactor: public NoiseModelFactor3<Pose3, Pose3, double> {
  private:

    typedef BetweenPoseScaleFactor This;
    typedef NoiseModelFactor3<Pose3, Pose3, double> Base;

    Pose3 measured_; /** The measurement */

  public:

    // shorthand for a smart pointer to a factor
    typedef typename boost::shared_ptr<BetweenPoseScaleFactor> shared_ptr;

    /** default constructor - only use for serialization */
    BetweenPoseScaleFactor() {}

    /** Constructor */
    BetweenPoseScaleFactor(Key key1, Key key2, Key scale_key, const Pose3& measured,
        const SharedNoiseModel& model = nullptr) :
      Base(model, key1, key2, scale_key), measured_(measured) {
    }

    virtual ~BetweenPoseScaleFactor() {}

    /// @return a deep copy of this factor
    virtual gtsam::NonlinearFactor::shared_ptr clone() const {
      return boost::static_pointer_cast<gtsam::NonlinearFactor>(
          gtsam::NonlinearFactor::shared_ptr(new This(*this))); }

    /** implement functions needed for Testable */

    /** print */
    virtual void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
      std::cout << s << "BetweenPoseScaleFactor("
          << keyFormatter(this->key1()) << ","
          << keyFormatter(this->key2()) << ","
          << keyFormatter(this->key3()) << ")\n";
      traits<Pose3>::Print(measured_, "  measured: ");
      this->noiseModel_->print("  noise model: ");
    }

    /** equals */
    virtual bool equals(const NonlinearFactor& expected, double tol=1e-9) const {
      const This *e =  dynamic_cast<const This*> (&expected);
      return e != nullptr && Base::equals(*e, tol) && traits<Pose3>::Equals(this->measured_, e->measured_, tol);
    }

    /** implement functions needed to derive from Factor */

    /** vector of errors */
    Vector evaluateError(const Pose3& p1, const Pose3& p2, const double& scale, boost::optional<Matrix&> H1 =
      boost::none, boost::optional<Matrix&> H2 = boost::none, boost::optional<Matrix&> H3 = boost::none) const {
      // scale measurement into real world space
      Pose3 measured_scaled(measured_.rotation(), measured_.translation() * scale);

      Pose3 hx = traits<Pose3>::Between(p1, p2, H1, H2); // h(x)/scale
      // manifold equivalent of h(x)-z -> log(z,h(x))
      typename traits<Pose3>::ChartJacobian::Jacobian Hlocal;
      Vector rval = traits<Pose3>::Local(measured_scaled, hx, boost::none, (H1 || H2) ? &Hlocal : 0);
      if (H1) {
        *H1 = Hlocal * (*H1);
      }
      if (H2) {
        *H2 = Hlocal * (*H2);
      }
      if (H3) {
        *H3 = Matrix::Zero(6, 1);
        // essentially the derivative of h - z*scale = -z, but
        // we first transform translation into global frame
        (*H3).block<3,1>(3,0) = -hx.rotation().inverse().rotate(measured_.translation());
      }
      return rval;
    }

    /** return the measured */
    const Pose3& measured() const {
      return measured_;
    }

    /** number of variables attached to this factor */
    std::size_t size() const {
      return 3;
    }

  private:

    /** Serialization function */
    friend class boost::serialization::access;
    template<class ARCHIVE>
    void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
      ar & boost::serialization::make_nvp("NoiseModelFactor3",
          boost::serialization::base_object<Base>(*this));
      ar & BOOST_SERIALIZATION_NVP(measured_);
    }

	  // Alignment, see https://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
	  enum { NeedsToAlign = (sizeof(Pose3) % 16) == 0 };
    public:
      GTSAM_MAKE_ALIGNED_OPERATOR_NEW_IF(NeedsToAlign)
  }; // \class BetweenPoseScaleFactor

  /// traits
  template<>
  struct traits<BetweenPoseScaleFactor> : public Testable<BetweenPoseScaleFactor> {};

  /**
   * Binary between constraint - forces between to a given value
   * This constraint requires the underlying type to a Lie type
   *
   */
  class BetweenPoseScaleConstraint : public BetweenPoseScaleFactor {
  public:
    typedef boost::shared_ptr<BetweenPoseScaleConstraint> shared_ptr;

    /** Syntactic sugar for constrained version */
    BetweenPoseScaleConstraint(const Pose3& measured, Key key1, Key key2, Key scale_key, double mu = 1000.0) :
      BetweenPoseScaleFactor(key1, key2, scale_key, measured,
                           noiseModel::Constrained::All(traits<Pose3>::GetDimension(measured), std::abs(mu)))
    {}

  private:

    /** Serialization function */
    friend class boost::serialization::access;
    template<class ARCHIVE>
    void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
      ar & boost::serialization::make_nvp("BetweenPoseScaleFactor",
          boost::serialization::base_object<BetweenPoseScaleFactor>(*this));
    }
  }; // \class BetweenConstraint

  /// traits
  template<>
  struct traits<BetweenPoseScaleConstraint> : public Testable<BetweenPoseScaleConstraint> {};

} /// namespace gtsam
