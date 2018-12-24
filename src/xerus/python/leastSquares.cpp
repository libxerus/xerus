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
 * @brief Definition of the python bindings of our least squares algorithms.
 */


#define NO_IMPORT_ARRAY
#include "misc.h"

void expose_leastSquaresAlgorithms() {
	VECTOR_TO_PY(PerformanceData::DataPoint, "PerfDataVector");
	{ scope perfdata =
		class_<PerformanceData>("PerformanceData")
			.def_readwrite("printProgress", &PerformanceData::printProgress)
			.def_readwrite("startTime", &PerformanceData::startTime)
			.def_readwrite("stopTime", &PerformanceData::stopTime)
			.def_readwrite("additionalInformation", &PerformanceData::additionalInformation)
			.add_property("data", +[](PerformanceData &_this){
				return _this.data;
			}, +[](PerformanceData &_this, std::vector<PerformanceData::DataPoint> &_newData){
				_this.data = _newData;
			})
			.add_property("errorFunction",
						  +[](PerformanceData &_this){ return _this.errorFunction; },
						  +[](PerformanceData &_this, PyObject *_f){
							  // TODO increase ref count for _f? also decrease it on overwrite?!
							  _this.errorFunction = [_f](const TTTensor &_x)->double{
								  return call<double>(_f, _x);
							};
						})
			.def(init<bool>())
			.def("start", &PerformanceData::start)
			.def("stop_timer", &PerformanceData::stop_timer)
			.def("continue_timer", &PerformanceData::continue_timer)
			.def("reset", &PerformanceData::reset)
			.def("get_elapsed_time", &PerformanceData::get_elapsed_time)
			.def("get_runtime", &PerformanceData::get_runtime)
			.def("add", +[](PerformanceData &_this, size_t _itr, const double _res, const TTTensor &_x, const size_t _flags) {
				_this.add(_itr, _res, _x, _flags);
			}, (arg("iteration"), arg("residual"), arg("x"), arg("flags")=0) )
			.def("add", +[](PerformanceData &_this, const double _res, const TTTensor &_x, size_t _flags) {
				_this.add(_res, _x, _flags);
			}, (arg("residual"), arg("x"), arg("flags")=0) )
			.def("__nonzero__", +[](PerformanceData &_this){ return bool(_this); })
			.def("dump_to_file", &PerformanceData::dump_to_file)
			.def("__iadd__", +[](PerformanceData &_this, const std::string &_s){
				_this << _s;
			})
			// TODO histogram
		;

		class_<PerformanceData::DataPoint>("DataPoint", no_init)
			.def_readonly("iteration", &PerformanceData::DataPoint::iteration)
			.def_readonly("elapsedTime", &PerformanceData::DataPoint::elapsedTime)
			.def_readonly("residuals", &PerformanceData::DataPoint::residuals)
			.def_readonly("error", &PerformanceData::DataPoint::error)
			.def_readonly("dofs", &PerformanceData::DataPoint::dofs)
			.def_readonly("flags", &PerformanceData::DataPoint::flags)
		;
	}

	class_<TTRetractionI>("TTRetractionI", init<const TTRetractionI &>());
	scope().attr("ALSRetractionI") = object(TTRetractionI(&ALSRetractionI));
	scope().attr("SubmanifoldRetractionI") = object(TTRetractionI(&SubmanifoldRetractionI));
	scope().attr("HOSVDRetractionI") = object(TTRetractionI(&HOSVDRetractionI));

	class_<TTRetractionII>("TTRetractionII", init<const TTRetractionII &>());
	scope().attr("ALSRetractionII") = object(TTRetractionII(&ALSRetractionII));
	scope().attr("SubmanifoldRetractionII") = object(TTRetractionII(&SubmanifoldRetractionII));
	scope().attr("HOSVDRetractionII") = object(TTRetractionII(&HOSVDRetractionII));

	class_<TTVectorTransport>("TTVectorTransport", init<const TTVectorTransport &>());
	scope().attr("ProjectiveVectorTransport") = object(TTVectorTransport(&ProjectiveVectorTransport));

	{ scope als_scope =
		class_<ALSVariant>("ALSVariant", init<uint, size_t, ALSVariant::LocalSolver, bool, optional<bool>>())
			.def(init<const ALSVariant&>())
			.def_readwrite("sites", &ALSVariant::sites)
			.def_readwrite("numHalfSweeps", &ALSVariant::numHalfSweeps)
			.def_readwrite("convergenceEpsilon", &ALSVariant::convergenceEpsilon)
			.def_readwrite("useResidualForEndCriterion", &ALSVariant::useResidualForEndCriterion)
			.def_readwrite("preserveCorePosition", &ALSVariant::preserveCorePosition)
			.def_readwrite("assumeSPD", &ALSVariant::assumeSPD)
			.add_property("localSolver",
						  +[](ALSVariant &_this){ return _this.localSolver; },
						  +[](ALSVariant &_this, ALSVariant::LocalSolver _s){ _this.localSolver = _s; })

			.def("__call__", +[](ALSVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, PerformanceData &_pd) {
				_this(_A, _x, _b, _pd);
			}, (arg("A"), arg("x"), arg("b"), arg("perfData")=NoPerfData) )

			.def("__call__", +[](ALSVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, value_t _eps, PerformanceData &_pd) {
				_this(_A, _x, _b, _eps, _pd);
			}, (arg("A"), arg("x"), arg("b"), arg("epsilon"), arg("perfData")=NoPerfData) )

			.def("__call__", +[](ALSVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, size_t _numHalfSweeps, PerformanceData &_pd) {
				_this(_A, _x, _b, _numHalfSweeps, _pd);
			}, (arg("A"), arg("x"), arg("b"), arg("numHalfSweeps"), arg("perfData")=NoPerfData) )

			.def("__call__", +[](ALSVariant &_this, TTTensor &_x, const TTTensor &_b, PerformanceData &_pd) {
				_this(_x, _b, _pd);
			}, (arg("x"), arg("b"), arg("perfData")=NoPerfData) )

			.def("__call__", +[](ALSVariant &_this, TTTensor &_x, const TTTensor &_b, value_t _eps, PerformanceData &_pd) {
				_this(_x, _b, _eps, _pd);
			}, (arg("x"), arg("b"), arg("epsilon"), arg("perfData")=NoPerfData) )

			.def("__call__", +[](ALSVariant &_this, TTTensor &_x, const TTTensor &_b, size_t _numHalfSweeps, PerformanceData &_pd) {
				_this(_x, _b, _numHalfSweeps, _pd);
			}, (arg("x"), arg("b"), arg("numHalfSweeps"), arg("perfData")=NoPerfData) )
		;
		class_<ALSVariant::LocalSolver>("LocalSolver", boost::python::no_init);
		als_scope.attr("lapack_solver") = object(ALSVariant::LocalSolver(&ALSVariant::lapack_solver));
		als_scope.attr("ASD_solver") = object(ALSVariant::LocalSolver(&ALSVariant::ASD_solver));
	}
	scope().attr("ALS") = object(ptr(&ALS));
	scope().attr("ALS_SPD") = object(ptr(&ALS_SPD));
	scope().attr("DMRG") = object(ptr(&DMRG));
	scope().attr("DMRG_SPD") = object(ptr(&DMRG_SPD));
	scope().attr("ASD") = object(ptr(&ASD));
	scope().attr("ASD_SPD") = object(ptr(&ASD_SPD));

	def("decomposition_als", &decomposition_als, (arg("x"), arg("b"), arg("epsilon")=EPSILON, arg("maxIterations")=1000));

	class_<GeometricCGVariant>("GeometricCGVariant", init<size_t, value_t, bool, TTRetractionI, TTVectorTransport>())
		.def(init<const GeometricCGVariant&>())
		.def_readwrite("numSteps", &GeometricCGVariant::numSteps)
		.def_readwrite("convergenceEpsilon", &GeometricCGVariant::convergenceEpsilon)
		.def_readwrite("assumeSymmetricPositiveDefiniteOperator", &GeometricCGVariant::assumeSymmetricPositiveDefiniteOperator)
		.add_property("retraction",
					  +[](GeometricCGVariant &_this){ return _this.retraction; },
					  +[](GeometricCGVariant &_this, TTRetractionI _r){ _this.retraction = _r; })
		.add_property("vectorTransport",
					  +[](GeometricCGVariant &_this){ return _this.vectorTransport; },
					  +[](GeometricCGVariant &_this, TTVectorTransport _transp){ _this.vectorTransport = _transp; })

		.def("__call__", +[](GeometricCGVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, PerformanceData &_pd) {
			_this(_A, _x, _b, _pd);
		}, (arg("A"), arg("x"), arg("b"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](GeometricCGVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, value_t _eps, PerformanceData &_pd) {
			_this(_A, _x, _b, _eps, _pd);
		}, (arg("A"), arg("x"), arg("b"), arg("epsilon"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](GeometricCGVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, size_t _numSteps, PerformanceData &_pd) {
			_this(_A, _x, _b, _numSteps, _pd);
		}, (arg("A"), arg("x"), arg("b"), arg("numSteps"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](GeometricCGVariant &_this, TTTensor &_x, const TTTensor &_b, PerformanceData &_pd) {
			_this(_x, _b, _pd);
		}, (arg("x"), arg("b"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](GeometricCGVariant &_this, TTTensor &_x, const TTTensor &_b, value_t _eps, PerformanceData &_pd) {
			_this(_x, _b, _eps, _pd);
		}, (arg("x"), arg("b"), arg("epsilon"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](GeometricCGVariant &_this, TTTensor &_x, const TTTensor &_b, size_t _numSteps, PerformanceData &_pd) {
			_this(_x, _b, _numSteps, _pd);
		}, (arg("x"), arg("b"), arg("numSteps"), arg("perfData")=NoPerfData) )
	;
	scope().attr("GeometricCG") = object(ptr(&GeometricCG));

	class_<SteepestDescentVariant>("SteepestDescentVariant", init<size_t, value_t, bool, TTRetractionII>())
		.def(init<const SteepestDescentVariant&>())
		.def_readwrite("numSteps", &SteepestDescentVariant::numSteps)
		.def_readwrite("convergenceEpsilon", &SteepestDescentVariant::convergenceEpsilon)
		.def_readwrite("assumeSymmetricPositiveDefiniteOperator", &SteepestDescentVariant::assumeSymmetricPositiveDefiniteOperator)
		.add_property("retraction",
					  +[](SteepestDescentVariant &_this){ return _this.retraction; },
					  +[](SteepestDescentVariant &_this, TTRetractionII _r){ _this.retraction = _r; })

		.def("__call__", +[](SteepestDescentVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, PerformanceData &_pd) {
			_this(_A, _x, _b, _pd);
		}, (arg("A"), arg("x"), arg("b"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](SteepestDescentVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, value_t _eps, PerformanceData &_pd) {
			_this(_A, _x, _b, _eps, _pd);
		}, (arg("A"), arg("x"), arg("b"), arg("epsilon"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](SteepestDescentVariant &_this, const TTOperator &_A, TTTensor &_x, const TTTensor &_b, size_t _numSteps, PerformanceData &_pd) {
			_this(_A, _x, _b, _numSteps, _pd);
		}, (arg("A"), arg("x"), arg("b"), arg("numSteps"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](SteepestDescentVariant &_this, TTTensor &_x, const TTTensor &_b, PerformanceData &_pd) {
			_this(_x, _b, _pd);
		}, (arg("x"), arg("b"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](SteepestDescentVariant &_this, TTTensor &_x, const TTTensor &_b, value_t _eps, PerformanceData &_pd) {
			_this(_x, _b, _eps, _pd);
		}, (arg("x"), arg("b"), arg("epsilon"), arg("perfData")=NoPerfData) )

		.def("__call__", +[](SteepestDescentVariant &_this, TTTensor &_x, const TTTensor &_b, size_t _numSteps, PerformanceData &_pd) {
			_this(_x, _b, _numSteps, _pd);
		}, (arg("x"), arg("b"), arg("numSteps"), arg("perfData")=NoPerfData) )
	;
	scope().attr("SteepestDescent") = object(ptr(&SteepestDescent));
}
