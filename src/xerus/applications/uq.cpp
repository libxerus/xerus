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
 * @brief Implementations of the utilities for uncertainity quantification.
 */

#include <xerus/applications/uq.h>

#include <xerus/misc/math.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/internal.h>

#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/legendre.hpp>

#ifdef _OPENMP
	#include <omp.h>
#endif

namespace xerus { namespace uq {
	
	Tensor polynomial_basis_evaluation(const double _value, const PolynomBasis _polyBasis, const size_t _basisSize) {
		if(_polyBasis == PolynomBasis::Hermite) {
			return hermite_evaluation(_value, _basisSize);
		} else {
			return legendre_evaluation(_value, _basisSize);
		}
	}
    
	Tensor hermite_evaluation(const double _v, const size_t _basisSize) {
		Tensor p({_basisSize});
		for (unsigned i = 0; i < _basisSize; ++i) {
			p[i] = std::sqrt(1/(/*std::sqrt(2*M_PI)**/boost::math::factorial<double>(i)))*boost::math::hermite(i, _v/std::sqrt(2))/std::pow(2.0, i/2.0);
		}
		return p; 
	}
	
	
	Tensor legendre_evaluation(const double _v, const size_t _basisSize) {
		Tensor p({_basisSize});
		for (unsigned i = 0; i < _basisSize; ++i) {
			p[i] = boost::math::legendre_p(int(i), _v);
		}
		return p;
	}
	
	
	std::tuple<Tensor, Tensor, Tensor> det_moments(const TTTensor& _x, const PolynomBasis _basisType) {
		REQUIRE(_x.corePosition == 0, "Invalid core position.");
		
		// M1		
		TTTensor m1TT = _x;
		
		while(m1TT.degree() > 1) {
			m1TT.fix_mode(1, 0);
		}
		double m1Factor = 1.0;
		if(_basisType == PolynomBasis::Hermite) {
// 			m1Factor = 1/(std::sqrt(2*M_PI));
// 			m1Factor = 1.0/10.0;
		}
		Tensor m1 = m1Factor*Tensor(m1TT);
		
		// M2
		Tensor spacialCmp = _x.get_component(0);
		spacialCmp.reinterpret_dimensions({spacialCmp.dimensions[1], spacialCmp.dimensions[2]});
		const auto sqrExpectation = contract(entrywise_product(spacialCmp, spacialCmp), xerus::Tensor::ones({spacialCmp.dimensions[1]}), 1);
		Tensor m2 = sqrExpectation;

		// M3
		Tensor m3(m2.dimensions);
		
		return std::make_tuple(m1, m2, m3);
	}
	
	
	std::tuple<Tensor, Tensor, Tensor> mc_moments(const TTTensor& _x, const PolynomBasis _basisType, const size_t _N) {
		Tensor m1({_x.dimensions[0]}), m2({_x.dimensions[0]}), m3({_x.dimensions[0]});
		
		const Tensor one = Tensor::ones({1});
		std::normal_distribution<double> hermiteDist(0.0, 1.0);
		std::uniform_real_distribution<double> legendreDist(-1.0, 1.0);
		
		for(size_t i = 0; i < _N; ++i) {
			Tensor p = one;
			for(size_t k = _x.degree()-1; k > 0; --k) {
				contract(p, _x.get_component(k), p, 1);
				contract(p, p, polynomial_basis_evaluation(_basisType == PolynomBasis::Hermite ? hermiteDist(misc::randomEngine) : legendreDist(misc::randomEngine), _basisType, _x.dimensions[k]), 1);
			}
			contract(p, _x.get_component(0), p, 1);
			p.reinterpret_dimensions({_x.dimensions[0]});
			
			const auto pSqr = entrywise_product(p,p);
			m1 += p;
			m2 += pSqr;
			m3 += entrywise_product(pSqr,p);
		}
		
		volatile double dN = double(_N); // TODO GCC bug? Removing volatile results in LTO to replace dN with ~0! 
		
		return std::make_tuple(m1/dN, m2/dN, m3/dN);
	}

	
	std::tuple<Tensor, Tensor, Tensor> mean_var_skew(const std::tuple<Tensor, Tensor, Tensor>& _moments) {
		const auto& mean = std::get<0>(_moments);
		const auto meanSqr = entrywise_product(mean, mean);
		const auto meanCube = entrywise_product(meanSqr, mean);
		
		// sigma^2 = m2 - mean^2
		const auto sigmaSqr = std::get<1>(_moments) - meanSqr;
		
		// skew = (m3 - 3*mean*sigma^2 - mean^3)/(sigma^3)
		auto skew = std::get<2>(_moments) - 3.0*entrywise_product(mean, sigmaSqr) - meanCube;
		skew.modify_entries([&sigmaSqr](value_t& _val, const size_t _idx){ _val /= std::pow(sigmaSqr[_idx], 1.5); });
		
		return std::make_tuple(std::get<0>(_moments), sigmaSqr, skew);
	}
	
	
	Tensor evaluate(const TTTensor& _x, const std::vector<double>& _parameters, const PolynomBasis _basisType) {
		REQUIRE(_x.degree() > 1, "IE");
		REQUIRE(_parameters.size()+1 == _x.degree(), "Invalid Parameters");
		Tensor p = Tensor::ones({1});
		for(size_t k = _x.degree()-1; k > 0; --k) {
			contract(p, _x.get_component(k), p, 1);
			contract(p, p, polynomial_basis_evaluation(_parameters[k-1], _basisType, _x.dimensions[k]), 1);
		}
		contract(p, _x.get_component(0), p, 1);
		p.reinterpret_dimensions({_x.dimensions[0]});
		return p;
	}
	
	
    void UQMeasurementSet::add(const std::vector<double>& _paramVec, const Tensor& _solution) {
		parameterVectors.push_back(_paramVec);
		solutions.push_back(_solution);
	}
	
	
	size_t UQMeasurementSet::size() const {
		return parameterVectors.size();
	}
	
	
	void UQMeasurementSet::clear() {
		parameterVectors.clear();
		solutions.clear();
	}
	
	
	Tensor sample_mean(const std::vector<Tensor>& _samples) {
		REQUIRE(_samples.size() > 0, "Need at least one measurment.");
		
		// Calc mean
		Tensor mean({_samples.front().size});
		for(const auto& samp : _samples) {
			mean += samp;
		}
		mean /= double(_samples.size());
		return mean;
	}
	
    
    TTTensor initial_guess(const Tensor& _mean, const UQMeasurementSet& _measurments, const PolynomBasis _polyBasis, const std::vector<size_t>& _dimensions) {
		REQUIRE(_measurments.parameterVectors.size() > 0, "Need at least one measurment.");
		REQUIRE(_measurments.parameterVectors.size() == _measurments.solutions.size(), "Invalid measurments.");
		REQUIRE(_dimensions.front() == _measurments.solutions.front().dimensions.front(), "Inconsitend measurments and dimensions.");
		
		TTTensor initalGuess(_dimensions);
		Tensor mean = _mean;
		
		// Set mean
		mean.reinterpret_dimensions({1, _dimensions[0], 1});
		initalGuess.set_component(0, mean);
		for(size_t k = 1; k < initalGuess.degree(); ++k) {
			initalGuess.set_component(k, Tensor::dirac({1, _dimensions[k], 1}, 0));
		}
		initalGuess.assume_core_position(0);
		
		mean.reinterpret_dimensions({_dimensions[0]});

		// Calc linear terms
		std::set<size_t> usedParams;
		for(size_t m = 0; m < _measurments.parameterVectors.size(); ++m) {
			const auto& paramVec = _measurments.parameterVectors[m];
			const auto& sol = _measurments.solutions[m];
			
			REQUIRE(paramVec.size()+1 == initalGuess.degree(), "Invalid parameter vector");
			
			// Find parameter number
			size_t p = initalGuess.degree();
			bool skip = false;
			for(size_t i = 0; i < paramVec.size(); ++i) {
				if(std::abs(paramVec[i]) > 0.0) {
					if(misc::contains(usedParams, i)) {
						LOG(info, "Skipping douplicate parameter " << i);
						skip = true;
						continue; 
					}
					REQUIRE(p == initalGuess.degree(), "Parameters contains several non-zero entries: " << paramVec);
					REQUIRE(!misc::contains(usedParams, i), "Parameters " << i << " appears twice!" << _measurments.parameterVectors);
					usedParams.emplace(i);
					p = i;
				}
			}
			if(skip) { continue; }
			REQUIRE(p != initalGuess.degree(), "Parameters contains no non-zero entry: " << paramVec);
			
			TTTensor linearTerm(_dimensions);
			Tensor tmp = (sol - mean)/paramVec[p]; // TODO
			tmp.reinterpret_dimensions({1, initalGuess.dimensions[0], 1});
			linearTerm.set_component(0, tmp);
			for(size_t k = 1; k < initalGuess.degree(); ++k) {
				if(k == p+1) {
					linearTerm.set_component(k, Tensor::dirac({1, _dimensions[k], 1}, 1));
				} else {
					REQUIRE(misc::hard_equal(paramVec[k-1], 0.0), "Invalid initial randVec.");
					linearTerm.set_component(k, Tensor::dirac({1, _dimensions[k], 1}, 0));
				}
			}
			linearTerm.assume_core_position(0);
			initalGuess += linearTerm;
		}
		
		initalGuess.round(1e-4);
		LOG(UQ_Inital_Guess, "Found linear terms for " << usedParams << ". Post roundign ranks: " << initalGuess.ranks());
		return initalGuess;
	}
	
} // namespace uq 
} // namespace xerus
