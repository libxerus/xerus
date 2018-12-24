// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2018 Benjamin Huber and Sebastian Wolf.
//
// Xerus is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
//
// Xerus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with Xerus. If not, see <http://www.gnu.org/licenses/>.
//
// For further information on Xerus visit https://libXerus.org
// or contact us at contact@libXerus.org.

/**
 * @file
 * @brief Definition of the python bindings of our recovery and completion algorithms.
 */


#define NO_IMPORT_ARRAY
#include "misc.h"
// using namespace uq;

void expose_recoveryAlgorithms() {
    // ------------------------------------------------------------- measurements

    class_<SinglePointMeasurementSet>("SinglePointMeasurementSet")
        .def(init<const SinglePointMeasurementSet&>())
        .def("get_position", +[](SinglePointMeasurementSet &_this, size_t _i){
            return _this.positions[_i];
        })
        .def("set_position", +[](SinglePointMeasurementSet &_this, size_t _i, std::vector<size_t> _pos){
            _this.positions[_i] = _pos;
        })
        .def("get_measuredValue", +[](SinglePointMeasurementSet &_this, size_t _i){
            return _this.measuredValues[_i];
        })
        .def("set_measuredValue", +[](SinglePointMeasurementSet &_this, size_t _i, value_t _val){
            _this.measuredValues[_i] = _val;
        })
        .def("add", &SinglePointMeasurementSet::add)
        .def("size", &SinglePointMeasurementSet::size)
        .def("degree", &SinglePointMeasurementSet::degree)
        .def("frob_norm", &SinglePointMeasurementSet::frob_norm)
        .def("sort", &SinglePointMeasurementSet::sort, arg("positionsOnly")=false)
        .def("measure", static_cast<void (SinglePointMeasurementSet::*)(const Tensor &)>(&SinglePointMeasurementSet::measure), arg("solution"))
        .def("measure", static_cast<void (SinglePointMeasurementSet::*)(const TensorNetwork &)>(&SinglePointMeasurementSet::measure), arg("solution"))
        .def("measure", +[](SinglePointMeasurementSet &_this, PyObject *_f) {
                            // TODO increase ref count for _f? also decrease it on overwrite?!
                            _this.measure([&_f](const std::vector<size_t> &pos)->double {
                                return call<double>(_f, pos);
                            });
                        })
        .def("test", static_cast<double (SinglePointMeasurementSet::*)(const Tensor &) const>(&SinglePointMeasurementSet::test), arg("solution"))
        .def("test", static_cast<double (SinglePointMeasurementSet::*)(const TensorNetwork &) const>(&SinglePointMeasurementSet::test), arg("solution"))
        .def("test", +[](SinglePointMeasurementSet &_this, PyObject *_f)->double {
                            // TODO increase ref count for _f? also decrease it on overwrite?!
                            return _this.test([&_f](const std::vector<size_t> &pos)->double {
                                return call<double>(_f, pos);
                            });
                        })


        .def("random",static_cast<SinglePointMeasurementSet (*)(size_t, const std::vector<size_t>&)>(&SinglePointMeasurementSet::random))
        .def("random",static_cast<SinglePointMeasurementSet (*)(size_t, const Tensor&)>(&SinglePointMeasurementSet::random))
        .def("random",static_cast<SinglePointMeasurementSet (*)(size_t, const TensorNetwork&)>(&SinglePointMeasurementSet::random))
        .def("random",+[](size_t n, const std::vector<size_t> &dim, PyObject *_f) {
                            // TODO increase ref count for _f? also decrease it on overwrite?!
                            return SinglePointMeasurementSet::random(n, dim, [&_f](const std::vector<size_t> &pos)->double {
                                return call<double>(_f, pos);
                            });
                        })
             .staticmethod("random")
    ;
    def("IHT", &IHT, (arg("x"), arg("measurements"), arg("perfData")=NoPerfData) );


    VECTOR_TO_PY(Tensor, "TensorVector");

    class_<RankOneMeasurementSet>("RankOneMeasurementSet")
        .def(init<const RankOneMeasurementSet&>())
        .def("get_position", +[](RankOneMeasurementSet &_this, size_t _i){
            return _this.positions[_i];
        })
        .def("set_position", +[](RankOneMeasurementSet &_this, size_t _i, std::vector<Tensor> _pos){
            _this.positions[_i] = _pos;
        })
        .def("get_measuredValue", +[](RankOneMeasurementSet &_this, size_t _i){
            return _this.measuredValues[_i];
        })
        .def("set_measuredValue", +[](RankOneMeasurementSet &_this, size_t _i, value_t _val){
            _this.measuredValues[_i] = _val;
        })
        .def("add", &RankOneMeasurementSet::add)
        .def("size", &RankOneMeasurementSet::size)
        .def("degree", &RankOneMeasurementSet::degree)
        .def("frob_norm", &RankOneMeasurementSet::frob_norm)
        .def("sort", &RankOneMeasurementSet::sort, arg("positionsOnly")=false)
        .def("normalize", &RankOneMeasurementSet::normalize)
        .def("measure", static_cast<void (RankOneMeasurementSet::*)(const Tensor &)>(&RankOneMeasurementSet::measure), arg("solution"))
        .def("measure", static_cast<void (RankOneMeasurementSet::*)(const TensorNetwork &)>(&RankOneMeasurementSet::measure), arg("solution"))
        .def("measure", +[](RankOneMeasurementSet &_this, PyObject *_f) {
                            // TODO increase ref count for _f? also decrease it on overwrite?!
                            _this.measure([&_f](const std::vector<Tensor> &pos)->double {
                                return call<double>(_f, pos);
                            });
                        })
        .def("test", static_cast<double (RankOneMeasurementSet::*)(const Tensor &) const>(&RankOneMeasurementSet::test), arg("solution"))
        .def("test", static_cast<double (RankOneMeasurementSet::*)(const TensorNetwork &) const>(&RankOneMeasurementSet::test), arg("solution"))
        .def("test", +[](RankOneMeasurementSet &_this, PyObject *_f)->double {
                            // TODO increase ref count for _f? also decrease it on overwrite?!
                            return _this.test([&_f](const std::vector<Tensor> &pos)->double {
                                return call<double>(_f, pos);
                            });
                        })


        .def("random",static_cast<RankOneMeasurementSet (*)(size_t, const std::vector<size_t>&)>(&RankOneMeasurementSet::random))
        .def("random",static_cast<RankOneMeasurementSet (*)(size_t, const Tensor&)>(&RankOneMeasurementSet::random))
        .def("random",static_cast<RankOneMeasurementSet (*)(size_t, const TensorNetwork&)>(&RankOneMeasurementSet::random))
        .def("random",+[](size_t n, const std::vector<size_t> &dim, PyObject *_f) {
                            // TODO increase ref count for _f? also decrease it on overwrite?!
                            return RankOneMeasurementSet::random(n, dim, [&_f](const std::vector<Tensor> &pos)->double {
                                return call<double>(_f, pos);
                            });
                        })
             .staticmethod("random")
    ;

    // ------------------------------------------------------------- ADF

    class_<ADFVariant>("ADFVariant", init<size_t, double, double>())
        .def(init<ADFVariant>())
        .def_readwrite("maxIterations", &ADFVariant::maxIterations)
        .def_readwrite("targetResidualNorm", &ADFVariant::targetResidualNorm)
        .def_readwrite("minimalResidualNormDecrease", &ADFVariant::minimalResidualNormDecrease)

        .def("__call__", +[](ADFVariant &_this, TTTensor& _x, const SinglePointMeasurementSet& _meas, PerformanceData& _pd){
            return _this(_x, _meas, _pd);
        }, (arg("x"), arg("measurements"), arg("perfData")=NoPerfData) )
        .def("__call__", +[](ADFVariant &_this, TTTensor& _x, const SinglePointMeasurementSet& _meas, const std::vector<size_t>& _maxRanks, PerformanceData& _pd){
            return _this(_x, _meas, _maxRanks, _pd);
        }, (arg("x"), arg("measurements"), arg("maxRanks"), arg("perfData")=NoPerfData) )

        .def("__call__", +[](ADFVariant &_this, TTTensor& _x, const RankOneMeasurementSet& _meas, PerformanceData& _pd){
            return _this(_x, _meas, _pd);
        }, (arg("x"), arg("measurements"), arg("perfData")=NoPerfData) )
        .def("__call__", +[](ADFVariant &_this, TTTensor& _x, const RankOneMeasurementSet& _meas, const std::vector<size_t>& _maxRanks, PerformanceData& _pd){
            return _this(_x, _meas, _maxRanks, _pd);
        }, (arg("x"), arg("measurements"), arg("maxRanks"), arg("perfData")=NoPerfData) )
    ;
    scope().attr("ADF") = object(ptr(&ADF));

    class_<uq::UQMeasurementSet>("UQMeasurementSet")
    .def(init<const uq::UQMeasurementSet&>())
    .def("add", &uq::UQMeasurementSet::add)
    ;

    VECTOR_TO_PY(std::vector<double>, "DoubleVectorVector");
    py_pair<std::vector<std::vector<double>>, std::vector<Tensor>>();

    //def("uq_adf", +[](const UQMeasurementSet& _measurments, const TTTensor& _guess) {
    //  return uq_adf(_measurments, _guess);
    //}, ( arg("measurments"), arg("guess")) );

    def("uq_ra_adf", +[](const uq::UQMeasurementSet& _measurements, const uq::PolynomBasis _basisType, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr){
            return uq::uq_ra_adf(_measurements, _basisType, _dimensions, _targetEps, _maxItr);
            }, (arg("measurements"), arg("polynombasis"), arg("dimensions"), arg("targeteps"), arg("maxitr"))
       );

    def("uq_ra_adf_iv", +[](TTTensor& _x, const uq::UQMeasurementSet& _measurements, const uq::PolynomBasis _basisType, const double _targetEps, const size_t _maxItr){
            return uq::uq_ra_adf_iv(_x, _measurements, _basisType, _targetEps, _maxItr);
            }, (arg("initial guess"), arg("measurements"), arg("polynombasis"), arg("targeteps"), arg("maxitr"))
       );

    def("uq_tt_evaluate", +[](const TTTensor& _x, const std::vector<double>& _parameters, const uq::PolynomBasis _basisType) {
            return uq::evaluate(_x, _parameters, _basisType);
            }, (arg("x"), arg("parameters"), arg("basisType"))
       );

    enum_<uq::PolynomBasis>("PolynomBasis")
        .value("Hermite", uq::PolynomBasis::Hermite)
        .value("Legendre", uq::PolynomBasis::Legendre)
    ;
}

