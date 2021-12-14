#include "misc.h"

void expose_recoveryAlgorithms(module& m) {
	// ------------------------------------------------------------- measurements

	class_<SinglePointMeasurementSet>(m, "SinglePointMeasurementSet")
		.def(init<>(), "constructs an empty measurement set")
		.def(init<const SinglePointMeasurementSet&>())
		.def_readwrite("positions", &SinglePointMeasurementSet::positions)
		.def_readwrite("measuredValues", &SinglePointMeasurementSet::measuredValues)
		/* .def("add", overload_cast<const std::vector<size_t>&, const value_t>(&SinglePointMeasurementSet::add)) */
		.def("add", static_cast<void (SinglePointMeasurementSet::*)(const std::vector<size_t>&, const value_t)>(&SinglePointMeasurementSet::add))
		/* .def("add", overload_cast<const std::vector<size_t>&, const value_t, const value_t>(&SinglePointMeasurementSet::add)) */
		.def("add", static_cast<void (SinglePointMeasurementSet::*)(const std::vector<size_t>&, const value_t, const value_t)>(&SinglePointMeasurementSet::add))
		.def("size", &SinglePointMeasurementSet::size)
		.def("order", &SinglePointMeasurementSet::order)
		.def("norm_2", &SinglePointMeasurementSet::norm_2)
		.def("sort", &SinglePointMeasurementSet::sort)
		/* .def("measure", overload_cast<const Tensor &>(&SinglePointMeasurementSet::measure), arg("solution")) */
		.def("measure", static_cast<void (SinglePointMeasurementSet::*)(const Tensor&)>(&SinglePointMeasurementSet::measure), arg("solution"))
		/* .def("measure", overload_cast<const TensorNetwork &>(&SinglePointMeasurementSet::measure), arg("solution")) */
		.def("measure", static_cast<void (SinglePointMeasurementSet::*)(const TensorNetwork&)>(&SinglePointMeasurementSet::measure), arg("solution"))
		.def("measure", +[](SinglePointMeasurementSet &_this, const std::function<double(const std::vector<size_t>)> _f) {
			_this.measure(_f);
		})
		/* .def("test", overload_cast<const Tensor &>(&SinglePointMeasurementSet::test, const_), arg("solution")) */
		.def("test", static_cast<double (SinglePointMeasurementSet::*)(const Tensor&) const>(&SinglePointMeasurementSet::test), arg("solution"))
		/* .def("test", overload_cast<const TensorNetwork &>(&SinglePointMeasurementSet::test, const_), arg("solution")) */
		.def("test", static_cast<double (SinglePointMeasurementSet::*)(const TensorNetwork&) const>(&SinglePointMeasurementSet::test), arg("solution"))
		.def("test", +[](SinglePointMeasurementSet &_this, const std::function<double(const std::vector<size_t>)> _f) -> double {
			return _this.test(_f);
		})

		.def_static("random",static_cast<SinglePointMeasurementSet (*)(size_t, const std::vector<size_t>&)>(&SinglePointMeasurementSet::random))
		.def_static("random",static_cast<SinglePointMeasurementSet (*)(size_t, const Tensor&)>(&SinglePointMeasurementSet::random))
		.def_static("random",static_cast<SinglePointMeasurementSet (*)(size_t, const TensorNetwork&)>(&SinglePointMeasurementSet::random))
		.def_static("random",+[](size_t n, const std::vector<size_t> &dim, const std::function<double(const std::vector<size_t>)> _f) {
			return SinglePointMeasurementSet::random(n, dim, _f);
		})
	;
	m.def("IHT", &IHT, arg("x"), arg("measurements"), arg("perfData")=NoPerfData);

	class_<RankOneMeasurementSet>(m, "RankOneMeasurementSet")
		.def(init<>(), "constructs an empty measurement set")
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
		.def("add", +[](RankOneMeasurementSet& _self, const std::vector<Tensor>& _position, const value_t _measuredValue) {
			_self.add(_position, _measuredValue);
		})
		.def("add", +[](RankOneMeasurementSet& _self, const std::vector<Tensor>& _position, const value_t _measuredValue, const value_t _weight) {
			_self.add(_position, _measuredValue, _weight);
		})
		.def("size", &RankOneMeasurementSet::size)
		.def("order", &RankOneMeasurementSet::order)
		.def("norm_2", &RankOneMeasurementSet::norm_2)
		.def("sort", &RankOneMeasurementSet::sort)
		.def("normalize", &RankOneMeasurementSet::normalize)
		/* .def("measure", overload_cast<const Tensor &>(&RankOneMeasurementSet::measure), arg("solution")) */
		.def("measure", static_cast<void (RankOneMeasurementSet::*)(const Tensor&)>(&RankOneMeasurementSet::measure), arg("solution"))
		/* .def("measure", overload_cast<const TensorNetwork &>(&RankOneMeasurementSet::measure), arg("solution")) */
		.def("measure", static_cast<void (RankOneMeasurementSet::*)(const TensorNetwork&)>(&RankOneMeasurementSet::measure), arg("solution"))
		.def("measure", +[](RankOneMeasurementSet &_this, const std::function<double(const std::vector<Tensor>)> _f) {
			_this.measure(_f);
		})
		/* .def("test", overload_cast<const Tensor &>(&RankOneMeasurementSet::test, const_), arg("solution")) */
		.def("test", static_cast<double (RankOneMeasurementSet::*)(const Tensor&) const>(&RankOneMeasurementSet::test), arg("solution"))
		/* .def("test", overload_cast<const TensorNetwork &>(&RankOneMeasurementSet::test, const_), arg("solution")) */
		.def("test", static_cast<double (RankOneMeasurementSet::*)(const TensorNetwork&) const>(&RankOneMeasurementSet::test), arg("solution"))
		.def("test", +[](RankOneMeasurementSet &_this, const std::function<double(const std::vector<Tensor>)> _f) -> double {
			return _this.test(_f);
		})

		.def_static("random",static_cast<RankOneMeasurementSet (*)(size_t, const std::vector<size_t>&)>(&RankOneMeasurementSet::random))
		.def_static("random",static_cast<RankOneMeasurementSet (*)(size_t, const Tensor&)>(&RankOneMeasurementSet::random))
		.def_static("random",static_cast<RankOneMeasurementSet (*)(size_t, const TensorNetwork&)>(&RankOneMeasurementSet::random))
		.def_static("random",+[](size_t n, const std::vector<size_t> &dim, const std::function<double(const std::vector<Tensor>)> _f) {
			return RankOneMeasurementSet::random(n, dim, _f);
		})
	;

	// ------------------------------------------------------------- ADF

	class_<ADFVariant>(m, "ADFVariant")
		.def(init<size_t, double, double>())
		.def(init<ADFVariant>())
		.def_readwrite("maxIterations", &ADFVariant::maxIterations)
		.def_readwrite("targetResidualNorm", &ADFVariant::targetRelativeResidual)
		.def_readwrite("minimalResidualDecrease", &ADFVariant::minimalResidualDecrease)

		.def("__call__", +[](ADFVariant &_this, TTTensor& _x, const SinglePointMeasurementSet& _meas, PerformanceData& _pd){
			return _this(_x, _meas, _pd);
		}, arg("x"), arg("measurements"), arg("perfData")=NoPerfData)
		.def("__call__", +[](ADFVariant &_this, TTTensor& _x, const SinglePointMeasurementSet& _meas, const std::vector<size_t>& _maxRanks, PerformanceData& _pd){
			return _this(_x, _meas, _maxRanks, _pd);
		}, arg("x"), arg("measurements"), arg("maxRanks"), arg("perfData")=NoPerfData)

		.def("__call__", +[](ADFVariant &_this, TTTensor& _x, const RankOneMeasurementSet& _meas, PerformanceData& _pd){
			return _this(_x, _meas, _pd);
		}, arg("x"), arg("measurements"), arg("perfData")=NoPerfData)
		.def("__call__", +[](ADFVariant &_this, TTTensor& _x, const RankOneMeasurementSet& _meas, const std::vector<size_t>& _maxRanks, PerformanceData& _pd){
			return _this(_x, _meas, _maxRanks, _pd);
		}, arg("x"), arg("measurements"), arg("maxRanks"), arg("perfData")=NoPerfData)
	;
	m.attr("ADF") = ADF;

	class_<uq::UQMeasurementSet>(m, "UQMeasurementSet")
		.def(init<>(), "constructs an empty measurement set")
		.def(init<const uq::UQMeasurementSet&>())
		.def("add", &uq::UQMeasurementSet::add)
	;

	m.def("uq_ra_adf", +[](const uq::UQMeasurementSet& _measurements, const uq::PolynomBasis _basisType, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr){
		return uq::uq_ra_adf(_measurements, _basisType, _dimensions, _targetEps, _maxItr);
		}, arg("measurements"), arg("polynombasis"), arg("dimensions"), arg("targeteps"), arg("maxitr")
	);

	m.def("uq_ra_adf", +[](const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr){
		return uq::uq_ra_adf(_positions, _solutions, _dimensions, _targetEps, _maxItr);
		}, arg("positions"), arg("solutions"), arg("dimensions"), arg("targeteps"), arg("maxitr")
	);

	m.def("uq_ra_adf", +[](const TTTensor& _x, const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr){
		return uq::uq_ra_adf(_x, _positions, _solutions, _dimensions, _targetEps, _maxItr);
		}, arg("x"), arg("positions"), arg("solutions"), arg("dimensions"), arg("targeteps"), arg("maxitr")
	);

	m.def("uq_ra_adf", +[](const std::vector<std::vector<Tensor>>& _positions, const std::vector<Tensor>& _solutions, const std::vector<double>& _weights, const std::vector<size_t>& _dimensions, const double _targetEps, const size_t _maxItr){
		return uq::uq_ra_adf(_positions, _solutions, _weights, _dimensions, _targetEps, _maxItr);
		}, arg("positions"), arg("solutions"), arg("weights"), arg("dimensions"), arg("targeteps"), arg("maxitr")
	);

	m.def("uq_ra_adf", +[](TTTensor& _x, const uq::UQMeasurementSet& _measurements, const uq::PolynomBasis _basisType, const double _targetEps, const size_t _maxItr){
		return uq::uq_ra_adf(_x, _measurements, _basisType, _targetEps, _maxItr);
		}, arg("initial guess"), arg("measurements"), arg("polynombasis"), arg("targeteps"), arg("maxitr")
	);

	m.def("uq_tt_evaluate", +[](const TTTensor& _x, const std::vector<double>& _parameters, const uq::PolynomBasis _basisType) {
		return uq::evaluate(_x, _parameters, _basisType);
		}, arg("x"), arg("parameters"), arg("basisType")
	);

	enum_<uq::PolynomBasis>(m, "PolynomBasis")
		.value("Hermite", uq::PolynomBasis::Hermite)
		.value("Legendre", uq::PolynomBasis::Legendre)
	;

	class_<uq::SALSA::State>(m, "uqSALSAState")
		.def_readonly("alpha", &uq::SALSA::State::alpha)
		.def_readonly("omega", &uq::SALSA::State::omega)
		.def_readonly("x", &uq::SALSA::State::x)
		.def_readonly("bestTrainingResidual", &uq::SALSA::State::trainingResidual)
		.def_readonly("bestValidationResidual", &uq::SALSA::State::validationResidual)
	;

	class_<uq::SALSA>(m, "uqSALSA")
		.def(init<TTTensor, std::vector<Tensor>, Tensor>())

		.def_readwrite("controlSetFraction", &uq::SALSA::controlSetFraction)
		.def_readwrite("targetResidual", &uq::SALSA::targetResidual)
		.def_readwrite("minDecrease", &uq::SALSA::minDecrease)
		.def_readwrite("maxSweeps", &uq::SALSA::maxSweeps)
		.def_readwrite("trackingPeriodLength", &uq::SALSA::trackingPeriodLength)
		.def_readwrite("maxStagnatingEpochs", &uq::SALSA::maxStagnatingEpochs)
		.def_readwrite("kmin", &uq::SALSA::kmin)
		.def_readwrite("maxRanks", &uq::SALSA::maxRanks)
		.def_readwrite("maxIRsteps", &uq::SALSA::maxIRsteps)
		.def_readwrite("IRtolerance", &uq::SALSA::IRtolerance)
		.def_readwrite("sparsityThreshold", &uq::SALSA::sparsityThreshold)
		.def_readwrite("fomega", &uq::SALSA::fomega)
		.def_readwrite("omegaFactor", &uq::SALSA::omegaFactor)
		.def_readwrite("falpha", &uq::SALSA::falpha)
		.def_readwrite("alphaFactor", &uq::SALSA::alphaFactor)
		.def_readwrite("basisWeights", &uq::SALSA::basisWeights)

		.def_readonly("initialResidual", &uq::SALSA::initialResidual)
		.def_readonly("bestIteration", &uq::SALSA::bestIteration)
		.def_readonly("bestState", &uq::SALSA::bestState)

		.def("run", &uq::SALSA::run)
	;
}

