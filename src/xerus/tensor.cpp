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
* @brief Implementation of the Tensor class.
*/

#include <xerus/tensor.h>
#include <xerus/misc/check.h>

#include <xerus/misc/containerSupport.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#include <xerus/blasLapackWrapper.h>
#include <xerus/cholmod_wrapper.h>
#include <xerus/sparseTimesFullContraction.h>
#include <xerus/index.h>
#include <xerus/tensorNetwork.h>

#include <xerus/cholmod_wrapper.h>

#include <fstream>
#include <iomanip>

namespace xerus {
	size_t Tensor::sparsityFactor = 4;
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	
	Tensor::Tensor(const Representation _representation) : Tensor(DimensionTuple({}), _representation) { } 
	
	
	Tensor::Tensor(DimensionTuple _dimensions, const Representation _representation, const Initialisation _init) 
		: dimensions(std::move(_dimensions)), size(misc::product(dimensions)), representation(_representation)
	{
		REQUIRE(size != 0, "May not create tensors with an dimension == 0.");
		
		if(representation == Representation::Dense) {
			denseData.reset(new value_t[size], internal::array_deleter_vt);
			if(_init == Initialisation::Zero) {
				misc::set_zero(denseData.get(), size);
			}
		} else {
			sparseData.reset(new std::map<size_t, value_t>());
		}
	}
	
	
	Tensor::Tensor(DimensionTuple _dimensions, std::unique_ptr<value_t[]>&& _data)
	: dimensions(std::move(_dimensions)), size(misc::product(dimensions)), representation(Representation::Dense), denseData(_data.release(), internal::array_deleter_vt) {
		REQUIRE(size != 0, "May not create tensors with an dimension == 0.");
	}
	
	
	Tensor::Tensor(DimensionTuple _dimensions, const std::function<value_t()>& _f) : Tensor(std::move(_dimensions), Representation::Dense, Initialisation::None) {
		value_t* const realData = denseData.get();
		for (size_t i=0; i < size; ++i) {
			realData[i] = _f();
		}
	}
	
	
	Tensor::Tensor(DimensionTuple _dimensions, const std::function<value_t(const size_t)>& _f) : Tensor(std::move(_dimensions), Representation::Dense, Initialisation::None) {
		value_t* const realData = denseData.get();
		for (size_t i=0; i < size; ++i) {
			realData[i] = _f(i);
		}
	}
	
	
	Tensor::Tensor(DimensionTuple _dimensions, const std::function<value_t(const MultiIndex&)>& _f) : Tensor(std::move(_dimensions), Representation::Dense, Initialisation::None) {
		value_t* const realData = denseData.get();
		MultiIndex multIdx(degree(), 0);
		size_t idx = 0;
		while (true) {
			realData[idx] = _f(multIdx);
			// Increasing indices
			idx++;
			size_t changingIndex = degree()-1;
			multIdx[changingIndex]++;
			while(multIdx[changingIndex] == dimensions[changingIndex]) {
				multIdx[changingIndex] = 0;
				changingIndex--;
				// Return on overflow 
				if(changingIndex >= degree()) { return; }
				multIdx[changingIndex]++;
			}
		}
	}
	
	
	Tensor::Tensor(DimensionTuple _dimensions, const size_t _N, const std::function<std::pair<size_t, value_t>(size_t, size_t)>& _f) : Tensor(std::move(_dimensions), Representation::Sparse, Initialisation::Zero) {
		REQUIRE(_N <= size, "Cannot create more non zero entries that the dimension of the Tensor.");
		for (size_t i = 0; i < _N; ++i) {
			std::pair<size_t, value_t> entry = _f(i, size);
			REQUIRE(entry.first < size, "Postion is out of bounds " << entry.first);
			REQUIRE(!misc::contains(*sparseData, entry.first), "Allready contained " << entry.first);
			sparseData->insert(std::move(entry));
		}
	}
	
	
	Tensor Tensor::ones(DimensionTuple _dimensions) {
		Tensor ret(std::move(_dimensions), Representation::Dense, Initialisation::None);
		value_t* const data = ret.get_dense_data();
		for(size_t i = 0; i < ret.size; ++i) {
			data[i] = 1.0;
		}
		return ret;
	}
		
	Tensor Tensor::identity(DimensionTuple _dimensions) {
		REQUIRE(_dimensions.size()%2 == 0, "Identity tensor must have even degree, here: " << _dimensions.size());
		const size_t d = _dimensions.size();
		
		Tensor ret(std::move(_dimensions), Representation::Sparse);
		MultiIndex position(d, 0);
		
		if(d == 0) {
			ret[position] = 1.0;
		} else {
			bool notMultiLevelBreak = true;
			while(notMultiLevelBreak) {
				ret[position] = 1.0;
				
				position[0]++; position[d/2]++;
				size_t node = 0;
				while(position[node] == std::min(ret.dimensions[node], ret.dimensions[d/2+node])) {
					position[node] = position[d/2+node] = 0;
					node++;
					if(node == d/2) {notMultiLevelBreak = false; break;}
					position[node]++; position[d/2+node]++;
				}
			}
		}
		
		return ret;
	}
	
	
	Tensor Tensor::kronecker(DimensionTuple _dimensions) {
		Tensor ret(std::move(_dimensions), Representation::Sparse);
		if(ret.degree() == 0) {
			ret[{}] = 1.0;
		} else {
			for(size_t i = 0; i < misc::min(ret.dimensions); ++i) {
				ret[MultiIndex(ret.degree(), i)] = 1.0;
			}
		}
		return ret;
	}
	
	
	Tensor Tensor::dirac(DimensionTuple _dimensions, const MultiIndex& _position) {
		Tensor ret(std::move(_dimensions), Representation::Sparse);
		ret[_position] = 1.0;
		return ret;
	}
	
	
	Tensor Tensor::dirac(DimensionTuple _dimensions, const size_t _position) {
		Tensor ret(std::move(_dimensions), Representation::Sparse);
		ret[_position] = 1.0;
		return ret;
	}
	
	
	Tensor Tensor::dense_copy() const {
		Tensor ret(*this);
		ret.use_dense_representation();
		return ret;
	}
	
	
	Tensor Tensor::sparse_copy() const {
		Tensor ret(*this);
		ret.use_sparse_representation();
		return ret;
	}
	
	Tensor& Tensor::operator=(const TensorNetwork& _network) {
		return operator=(Tensor(_network));
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Information - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	size_t Tensor::degree() const {
		return dimensions.size();
	}
	
	
	bool Tensor::has_factor() const {
// 		return std::abs(1.0-factor) > std::numeric_limits<double>::epsilon();
		#pragma GCC diagnostic push
			#pragma GCC diagnostic ignored "-Wfloat-equal"
			return (factor != 1.0);
		#pragma GCC diagnostic pop
	}
	
	
	bool Tensor::is_dense() const {
		INTERNAL_CHECK((representation == Representation::Dense && denseData && !sparseData) || (representation == Representation::Sparse && sparseData && !denseData), "Internal Error: " << bool(representation) << bool(denseData) << bool(sparseData));
		return representation == Representation::Dense;
	}
	
	
	bool Tensor::is_sparse() const {
		INTERNAL_CHECK((representation == Representation::Dense && denseData && !sparseData) || (representation == Representation::Sparse && sparseData && !denseData), "Internal Error: " << bool(representation) << bool(denseData) << bool(sparseData));
		return representation == Representation::Sparse;
	}
	
	
	size_t Tensor::sparsity() const {
		if(is_sparse()) {
			return sparseData->size();
		}
		return size;
	}
	
	
	size_t Tensor::count_non_zero_entries(const value_t _eps) const {
		if(is_dense()) {
			size_t count = 0;
			for(size_t i = 0; i < size; ++i) {
				if(std::abs(denseData.get()[i]) > _eps ) { count++; }
			}
			return count;
		}
		size_t count = 0;
		for(const auto& entry : *sparseData) {
			if(std::abs(entry.second) > _eps) { count++; } 
		}
		return count;
	}
	
	
	bool Tensor::all_entries_valid() const {
		if(!std::isfinite(factor)) { return false; } 
		
		if(is_dense()) {
			for(size_t i = 0; i < size; ++i) {
				if(!std::isfinite(denseData.get()[i])) { return false; } 
			}
		} else {
			for(const auto& entry : *sparseData) {
				if(!std::isfinite(entry.second)) {return false; } 
			}
		}
		return true;
	}
	
	
	size_t Tensor::reorder_cost() const {
		return is_sparse() ? 10*sparsity() : size;
	}
	
	
	value_t Tensor::one_norm() const {
		if(is_dense()) {
			return std::abs(factor)*blasWrapper::one_norm(denseData.get(), size);
		} 
		value_t norm = 0;
		for(const auto& entry : *sparseData) {
			norm += std::abs(entry.second);
		}
		return std::abs(factor)*norm;
	}
	
	value_t Tensor::frob_norm() const {
		if(is_dense()) {
			return std::abs(factor)*blasWrapper::two_norm(denseData.get(), size);
		} 
		value_t norm = 0;
		for(const auto& entry : *sparseData) {
			norm += misc::sqr(entry.second);
		}
		return std::abs(factor)*sqrt(norm);
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	Tensor& Tensor::operator+=(const Tensor& _other) {
		plus_minus_equal<1>(*this, _other);
		return *this;
	}
	
	
	Tensor& Tensor::operator-=(const Tensor& _other) {
		plus_minus_equal<-1>(*this, _other);
		return *this;
	}
	
	
	Tensor& Tensor::operator*=(const value_t _factor) {
		factor *= _factor;
		return *this;
	}
	
	
	Tensor& Tensor::operator/=(const value_t _divisor) {
		factor /= _divisor;
		return *this;
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Access - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	value_t& Tensor::operator[](const size_t _position) {
		REQUIRE(_position < size, "Position " << _position << " does not exist in Tensor of dimensions " << dimensions);
		
		ensure_own_data_and_apply_factor();
		
		if(is_dense()) {
			return denseData.get()[_position];
		} else {
			value_t& result = (*sparseData)[_position];
			use_dense_representation_if_desirable();
			if (is_dense()) {
				return denseData.get()[_position];
			}
			return result;
		}
	}
	

	value_t Tensor::operator[](const size_t _position) const {
		REQUIRE(_position < size, "Position " << _position << " does not exist in Tensor of dimensions " << dimensions);
		
		if(is_dense()) {
			return factor*denseData.get()[_position];
		} else {
			const std::map<size_t, value_t>::const_iterator entry = sparseData->find(_position);
			if(entry == sparseData->end()) {
				return 0.0;
			}
			return factor*entry->second;
		}
	}
	
	
	value_t& Tensor::operator[](const MultiIndex& _positions) {
		return operator[](multiIndex_to_position(_positions, dimensions));
	}
	
	
	value_t Tensor::operator[](const MultiIndex& _positions) const {
		return operator[](multiIndex_to_position(_positions, dimensions));
	}
	
	
	value_t& Tensor::at(const size_t _position) {
		REQUIRE(_position < size, "Position " << _position << " does not exist in Tensor of dimensions " << dimensions);
		REQUIRE(!has_factor(), "at() must not be called if there is a factor.");
		REQUIRE((is_dense() && denseData.unique()) || (is_sparse() && sparseData.unique()) , "Data must be unique to call at().");
		
		if(is_dense()) {
			return denseData.get()[_position];
		} 
		value_t& result = (*sparseData)[_position];
		use_dense_representation_if_desirable();
		if (is_dense()) {
			return denseData.get()[_position];
		}
		
		return result;
	}
	
	
	value_t Tensor::cat(const size_t _position) const {
		REQUIRE(_position < size, "Position " << _position << " does not exist in Tensor of dimensions " << dimensions);
		REQUIRE(!has_factor(), "at() must not be called if there is a factor.");
		
		if(is_dense()) {
			return denseData.get()[_position];
		} 
		const std::map<size_t, value_t>::const_iterator entry = sparseData->find(_position);
		if(entry == sparseData->end()) {
			return 0.0;
		}
		return entry->second;
	}
	
	
	value_t* Tensor::get_dense_data() {
		use_dense_representation();
		ensure_own_data_and_apply_factor();
		return denseData.get();
	}
	
	
	value_t* Tensor::get_unsanitized_dense_data() {
		REQUIRE(is_dense(), "Unsanitized dense data requested, but representation is not dense!");
		return denseData.get();
	}
	
	
	const value_t* Tensor::get_unsanitized_dense_data() const  {
		REQUIRE(is_dense(), "Unsanitized dense data requested, but representation is not dense!");
		return denseData.get();
	}
	
	
	value_t* Tensor::override_dense_data()  {
		factor = 1.0;
		if(!denseData.unique()) {
			sparseData.reset();
			denseData.reset(new value_t[size], internal::array_deleter_vt);
			representation = Representation::Dense;
		}
		return denseData.get();
	}
	
	
	const std::shared_ptr<value_t>& Tensor::get_internal_dense_data() {
		REQUIRE(is_dense(), "Internal dense data requested, but representation is not dense!");
		return denseData;
	}
	
	
	std::map<size_t, value_t>& Tensor::get_sparse_data() {
		CHECK(is_sparse(), warning, "Request for sparse data although the Tensor is not sparse.");
		use_sparse_representation();
		ensure_own_data_and_apply_factor();
		return *sparseData.get();
	}
	
	
	std::map<size_t, value_t>& Tensor::get_unsanitized_sparse_data() {
		REQUIRE(is_sparse(), "Unsanitized sparse data requested, but representation is not sparse!");
		return *sparseData.get();
	}
	
	
	const std::map<size_t, value_t>& Tensor::get_unsanitized_sparse_data() const  {
		REQUIRE(is_sparse(), "Unsanitized sparse data requested, but representation is not sparse!");
		return *sparseData.get();
	}
	
	
	std::map<size_t, value_t>& Tensor::override_sparse_data() {
		factor = 1.0;
		if(sparseData.unique()) {
			INTERNAL_CHECK(is_sparse(), "Internal Error");
			sparseData->clear();
		} else {
			denseData.reset();
			sparseData.reset(new std::map<size_t, value_t>());
			representation = Representation::Sparse;
		}
		
		return *sparseData.get();
	}
	
	
	const std::shared_ptr<std::map<size_t, value_t>>& Tensor::get_internal_sparse_data() {
		REQUIRE(is_sparse(), "Internal sparse data requested, but representation is not sparse!");
		return sparseData;
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Indexing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	internal::IndexedTensor<Tensor> Tensor::operator()(const std::vector<Index>&  _indices) {
		return internal::IndexedTensor<Tensor>(this, _indices, false);
	}
	
	
	internal::IndexedTensor<Tensor> Tensor::operator()(      std::vector<Index>&& _indices) {
		return internal::IndexedTensor<Tensor>(this, std::move(_indices), false);
	}
	
	
	internal::IndexedTensorReadOnly<Tensor> Tensor::operator()(const std::vector<Index>&  _indices) const {
		return internal::IndexedTensorReadOnly<Tensor>(this, _indices);
	}
	
	
	internal::IndexedTensorReadOnly<Tensor> Tensor::operator()(      std::vector<Index>&& _indices) const {
		return internal::IndexedTensorReadOnly<Tensor>(this, std::move(_indices));
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Modififiers - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	void Tensor::reset(DimensionTuple _newDim, const Representation _representation, const Initialisation _init) {
		const size_t oldDataSize = size;
		dimensions = std::move(_newDim);
		size = misc::product(dimensions);
		factor = 1.0;
		
		if(_representation == Representation::Dense) {
			if(representation == Representation::Dense) {
				if(oldDataSize != size || !denseData.unique()) {
					denseData.reset(new value_t[size], internal::array_deleter_vt);
				}
			} else {
				sparseData.reset();
				denseData.reset(new value_t[size], internal::array_deleter_vt);
				representation = _representation;
			}
			
			if(_init == Initialisation::Zero) {
				memset(denseData.get(), 0, size*sizeof(value_t));
			}
		} else {
			if(representation == Representation::Sparse) {
				if(sparseData.unique()) {
					sparseData->clear();
				} else {
					sparseData.reset(new std::map<size_t, value_t>());
				}
			} else {
				denseData.reset();
				sparseData.reset(new std::map<size_t, value_t>());
				representation = _representation;
			}
		}
	}
	
	
	void Tensor::reset(DimensionTuple _newDim, const Initialisation _init) {
		const size_t oldDataSize = size;
		dimensions = std::move(_newDim);
		size = misc::product(dimensions);
		factor = 1.0;
		
		if(representation == Representation::Dense) {
			if(oldDataSize != size || !denseData.unique()) {
				denseData.reset(new value_t[size], internal::array_deleter_vt);
			}
			
			if(_init == Initialisation::Zero) {
				memset(denseData.get(), 0, size*sizeof(value_t));
			}
		} else {
			if(sparseData.unique()) {
				sparseData->clear();
			} else {
				sparseData.reset(new std::map<size_t, value_t>());
			}
		}
	}
	
	void Tensor::reset() {
		dimensions.clear();
		factor = 1.0;
		
		if(representation == Representation::Dense) {
			if(size != 1 || !denseData.unique()) {
				denseData.reset(new value_t[1], internal::array_deleter_vt);
			}
			*denseData.get() = 0.0;
		} else {
			if(sparseData.unique()) {
				sparseData->clear();
			} else {
				sparseData.reset(new std::map<size_t, value_t>());
			}
		}
		size = 1;
	}
	
	
	void Tensor::reset(DimensionTuple _newDim, const std::shared_ptr<value_t>& _newData) {
		dimensions = std::move(_newDim);
		size = misc::product(dimensions);
		factor = 1.0;
		
		if(representation == Representation::Sparse) {
			sparseData.reset();
			representation = Representation::Dense;
		}
		
		denseData = _newData;
	}
	
	
	void Tensor::reset(DimensionTuple _newDim, std::unique_ptr<value_t[]>&& _newData) {
		dimensions = std::move(_newDim);
		size = misc::product(dimensions);
		factor = 1.0;
		
		if(representation == Representation::Sparse) {
			sparseData.reset();
			representation = Representation::Dense;
		}
		
		denseData.reset(_newData.release(), internal::array_deleter_vt);
	}
	
	void Tensor::reset(DimensionTuple _newDim, std::map<size_t, value_t>&& _newData) {
		dimensions = std::move(_newDim);
		size = misc::product(dimensions);
		factor = 1.0;
		
		if(representation == Representation::Dense) {
			denseData.reset();
			representation = Representation::Sparse;
		}
		
		std::shared_ptr<std::map<size_t, value_t>> newD(new std::map<size_t, value_t>(std::move(_newData)));
		sparseData = std::move(newD);
	}
		
	void Tensor::reinterpret_dimensions(DimensionTuple _newDimensions) {
		REQUIRE(misc::product(_newDimensions) == size, "New dimensions must not change the size of the tensor in reinterpretation: " << misc::product(_newDimensions) << " != " << size);
		dimensions = std::move(_newDimensions); // NOTE pass-by-value
	}
	
	
	void Tensor::resize_mode(const size_t _mode, const size_t _newDim, size_t _cutPos) {
		REQUIRE(_mode < degree(), "Can't resize mode " << _mode << " as the tensor is only order " << degree());

		if (dimensions[_mode] == _newDim) { return; }  // Trivial case: Nothing to do

		const size_t oldDim = dimensions[_mode];
		_cutPos = std::min(_cutPos, oldDim);
		
		REQUIRE(_newDim > 0, "Dimension must be larger than 0! Is " << _newDim);
		REQUIRE(_newDim > oldDim || _cutPos >= oldDim -_newDim, "Cannot remove " << oldDim -_newDim << " slates starting (exclusivly) at position " << _cutPos);
		
		const size_t dimStepSize = misc::product(dimensions, _mode+1, degree());
		const size_t oldStepSize = oldDim*dimStepSize;
		const size_t newStepSize = _newDim*dimStepSize;
		const size_t blockCount = size / oldStepSize; //  == misc::product(dimensions, 0, _n);
		const size_t newsize = blockCount*newStepSize;
		
		if(is_dense()) {
			std::unique_ptr<value_t[]> tmpData(new value_t[newsize]);
			
			if (_newDim > oldDim) { // Add new slates
				const size_t insertBlockSize = (_newDim-oldDim)*dimStepSize;
				if(_cutPos == 0) {
					for ( size_t i = 0; i < blockCount; ++i ) {
						value_t* const currData = tmpData.get()+i*newStepSize;
						misc::set_zero(currData, insertBlockSize);
						misc::copy(currData+insertBlockSize, denseData.get()+i*oldStepSize, oldStepSize);
					}
				} else if(_cutPos == oldDim) {
					for ( size_t i = 0; i < blockCount; ++i ) {
						value_t* const currData = tmpData.get()+i*newStepSize;
						misc::copy(currData, denseData.get()+i*oldStepSize, oldStepSize);
						misc::set_zero(currData+oldStepSize, insertBlockSize);
					}
				} else {
					const size_t preBlockSize = _cutPos*dimStepSize;
					const size_t postBlockSize = (oldDim-_cutPos)*dimStepSize;
					for (size_t i = 0; i < blockCount; ++i) {
						value_t* const currData = tmpData.get()+i*newStepSize;
						misc::copy(currData, denseData.get()+i*oldStepSize, preBlockSize);
						misc::set_zero(currData+preBlockSize, insertBlockSize);
						misc::copy(currData+preBlockSize+insertBlockSize, denseData.get()+i*oldStepSize+preBlockSize, postBlockSize);
					}
				}
			} else { // Remove slates
				if (_cutPos < oldDim) {
					const size_t removedSlates = (oldDim-_newDim);
					const size_t removedBlockSize = removedSlates*dimStepSize;
					const size_t preBlockSize = (_cutPos-removedSlates)*dimStepSize;
					const size_t postBlockSize = (oldDim-_cutPos)*dimStepSize;
					
					INTERNAL_CHECK(removedBlockSize+preBlockSize+postBlockSize == oldStepSize && preBlockSize+postBlockSize == newStepSize, "IE");
					
					for (size_t i = 0; i < blockCount; ++i) {
						value_t* const currData = tmpData.get()+i*newStepSize;
						misc::copy(currData, denseData.get()+i*oldStepSize, preBlockSize);
						misc::copy(currData+preBlockSize, denseData.get()+i*oldStepSize+preBlockSize+removedBlockSize, postBlockSize);
					}
				} else {
					for (size_t i = 0; i < blockCount; ++i) {
						misc::copy(tmpData.get()+i*newStepSize, denseData.get()+i*oldStepSize, newStepSize);
					}
				}
			}
			denseData.reset(tmpData.release(), internal::array_deleter_vt);
		
		} else {
			std::unique_ptr<std::map<size_t, value_t>> tmpData(new std::map<size_t, value_t>());
			
			if (_newDim > oldDim) { // Add new slates
				const size_t slatesAdded = _newDim-oldDim;
				for(const auto& entry : *sparseData.get()) {
					// Decode the position as i*oldStepSize + j*DimStepSize + k
					const size_t k = entry.first%dimStepSize;
					const size_t j = (entry.first%oldStepSize)/dimStepSize;
					const size_t i = entry.first/oldStepSize;
					
					if(j < _cutPos) { // Entry remains at current position j
						tmpData->emplace(i*newStepSize+j*dimStepSize+k, entry.second);
					} else { // Entry moves to position j+slatesAdded
						tmpData->emplace(i*newStepSize+(j+slatesAdded)*dimStepSize+k, entry.second);
					}
				}
			} else { // Remove slates
				const size_t slatesRemoved = oldDim-_newDim;
				for(const auto& entry : *sparseData.get()) {
					// Decode the position as i*oldStepSize + j*DimStepSize + k
					const size_t k = entry.first%dimStepSize;
					const size_t j = (entry.first%oldStepSize)/dimStepSize;
					const size_t i = entry.first/oldStepSize;
					
					if(j < _cutPos-slatesRemoved) { // Entry remains at current position j
						tmpData->emplace(i*newStepSize+j*dimStepSize+k, entry.second);
					} else if(j >= _cutPos) { // Entry advances in position j
						tmpData->emplace(i*newStepSize+(j-slatesRemoved)*dimStepSize+k, entry.second);
					}
				}
			}
			sparseData.reset(tmpData.release());
		}
		
		dimensions[_mode] = _newDim;
		size = newsize;
	}
	
	
	void Tensor::fix_mode(const size_t _mode, const size_t _slatePosition) {
		REQUIRE(_slatePosition < dimensions[_mode], "The given slatePosition must be smaller than the corresponding dimension. Here " << _slatePosition << " >= " << dimensions[_mode] << ", dim = " << dimensions << "=" << size << " mode " << _mode);
		
		const size_t stepCount = misc::product(dimensions, 0, _mode);
		const size_t blockSize = misc::product(dimensions, _mode+1, degree());
		const size_t stepSize = dimensions[_mode]*blockSize;
		const size_t slateOffset = _slatePosition*blockSize;
		
		if(is_dense()) {
			std::unique_ptr<value_t[]> tmpData(new value_t[stepCount*blockSize]);
			
			// Copy data
			for(size_t i = 0; i < stepCount; ++i) {
				misc::copy(tmpData.get()+i*blockSize, denseData.get()+i*stepSize+slateOffset, blockSize);
			}
			
			denseData.reset(tmpData.release(), &internal::array_deleter_vt);
		} else {
			std::unique_ptr<std::map<size_t, value_t>> tmpData(new std::map<size_t, value_t>());
			
			for(const auto& entry : *sparseData.get()) {
				// Decode the position as i*stepSize + j*blockSize + k
				const size_t k = entry.first%blockSize;
				const size_t j = (entry.first%stepSize)/blockSize;
				const size_t i = entry.first/stepSize;
				
				if(j == _slatePosition) {
					tmpData->emplace(i*blockSize+k, entry.second);
				}
			}
			
			sparseData.reset(tmpData.release());
		}
		
		// Adjust dimensions
		dimensions.erase(dimensions.begin()+long(_mode));
		size = stepCount*blockSize;
	}
	
	
	void Tensor::remove_slate(const size_t _mode, const size_t _pos) {
		REQUIRE(_mode < degree(), "");
		REQUIRE(_pos < dimensions[_mode], _pos << " " << dimensions[_mode]);
		REQUIRE(dimensions[_mode] > 1, "");
		
		resize_mode(_mode, dimensions[_mode]-1, _pos+1);
	}
	
	
	void Tensor::perform_trace(size_t _firstMode, size_t _secondMode) {
		REQUIRE(_firstMode != _secondMode, "Given indices must not coincide");
		REQUIRE(dimensions[_firstMode] == dimensions[_secondMode], "Dimensions of trace indices must coincide.");
		
		if(_firstMode > _secondMode) { std::swap(_firstMode, _secondMode); }
		
		
		
		const size_t front = misc::product(dimensions, 0, _firstMode);
		const size_t mid = misc::product(dimensions, _firstMode+1, _secondMode);
		const size_t back = misc::product(dimensions, _secondMode+1, degree());
		const size_t traceDim = dimensions[_firstMode];
		const size_t frontStepSize = traceDim*mid*traceDim*back;
		const size_t traceStepSize = mid*traceDim*back+back;
		const size_t midStepSize = traceDim*back;
		
		size = front*mid*back;
		
		if(is_dense()) {
			std::unique_ptr<value_t[]> newData(new value_t[size]);
			misc::set_zero(newData.get(), size);
			
			for(size_t f = 0; f < front; ++f) {
				for(size_t t = 0; t < traceDim; ++t) {
					for(size_t m = 0; m < mid; ++m) {
						misc::add_scaled(newData.get()+(f*mid+m)*back, factor, denseData.get()+f*frontStepSize+t*traceStepSize+m*midStepSize, back);
					}
				}
			}
			
			denseData.reset(newData.release(), internal::array_deleter_vt);
		} else {
			std::unique_ptr<std::map<size_t, value_t>> newData( new std::map<size_t, value_t>());
			
			for(const auto& entry : *sparseData) {
				size_t pos = entry.first;
				const size_t backIdx = pos%back;
				pos /= back;
				const size_t traceBackIdx = pos%traceDim;
				pos /= traceDim;
				const size_t midIdx = pos%mid;
				pos /= mid;
				const size_t traceFrontIdx = pos%traceDim;
				pos /= traceDim;
				const size_t frontIdx = pos;
				
				if(traceFrontIdx == traceBackIdx) {
					(*newData)[(frontIdx*mid + midIdx)*back + backIdx] += factor*entry.second;
				}
			}
			
			sparseData.reset(newData.release());
		}
		
		dimensions.erase(dimensions.begin()+_secondMode);
		dimensions.erase(dimensions.begin()+_firstMode);
		factor = 1.0;
	}
	
	
	void Tensor::modify_diagonal_entries(const std::function<void(value_t&)>& _f) {
		ensure_own_data_and_apply_factor();
		
		if(degree() == 0) {
			_f(at(0)); 
		} else {
			size_t stepSize = 1;
			for(size_t i = 1; i < degree(); ++i) {
				stepSize *= dimensions[i];
				stepSize += 1;
			}
			
			const size_t numDiags = misc::min(dimensions);
			
			for(size_t i = 0; i < numDiags; ++i){
				_f(at(i*stepSize));
			}
		}
	}
	
	
	void Tensor::modify_diagonal_entries(const std::function<void(value_t&, const size_t)>& _f) {
		ensure_own_data_and_apply_factor();
		
		if(degree() == 0) {
			_f(at(0), 0); 
		} else {
			size_t stepSize = 1;
			for(size_t i = 1; i < degree(); ++i) {
				stepSize *= dimensions[i];
				stepSize += 1;
			}
			
			const size_t numDiags = misc::min(dimensions);
			
			for(size_t i = 0; i < numDiags; ++i){
				_f(at(i*stepSize), i);
			}
		}
	}
	
	
	void Tensor::modify_entries(const std::function<void(value_t&)>& _f) {
		ensure_own_data_and_apply_factor();
		if(is_dense()) {
			for(size_t i = 0; i < size; ++i) { _f(at(i)); }
		} else {
			for(size_t i = 0; i < size; ++i) {
				value_t val = cat(i);
				const value_t oldVal = val;
				_f(val);
				if(misc::hard_not_equal(val, 0.0)) {
					at(i) = val;
				} else if( misc::hard_not_equal(oldVal, 0.0)) {
					IF_CHECK( size_t succ =) sparseData->erase(i);
					INTERNAL_CHECK(succ == 1, "Internal Error");
				}
			}
		}
	}
	

	void Tensor::modify_entries(const std::function<void(value_t&, const size_t)>& _f) {
		ensure_own_data_and_apply_factor();
		if(is_dense()) {
			for(size_t i = 0; i < size; ++i) { _f(at(i), i); }
		} else {
			for(size_t i = 0; i < size; ++i) {
				value_t val = cat(i);
				const value_t oldVal = val;
				_f(val, i);
				if(misc::hard_not_equal(val, 0.0)) {
					at(i) = val;
				} else if( misc::hard_not_equal(oldVal, 0.0)) {
					IF_CHECK( size_t succ =) sparseData->erase(i);
					INTERNAL_CHECK(succ == 1, "Internal Error");
				}
			}
		}
	}
	
	
	void Tensor::modify_entries(const std::function<void(value_t&, const MultiIndex&)>& _f) {
		ensure_own_data_and_apply_factor();
		
		MultiIndex multIdx(degree(), 0);
		size_t idx = 0;
		while (true) {
			if(is_dense()) {
				_f(at(idx), multIdx);
			} else {
				value_t val = cat(idx);
				const value_t oldVal = val;
				_f(val, multIdx);
				if(misc::hard_not_equal(val, 0.0)) {
					at(idx) = val;
				} else if( misc::hard_not_equal(oldVal, 0.0)) {
					IF_CHECK( size_t succ =) sparseData->erase(idx);
					INTERNAL_CHECK(succ == 1, "Internal Error");
				}
			}
			
			// Increase indices
			idx++;
			size_t changingIndex = degree()-1;
			multIdx[changingIndex]++;
			while(multIdx[changingIndex] == dimensions[changingIndex]) {
				multIdx[changingIndex] = 0;
				changingIndex--;
				// Return on overflow 
				if(changingIndex >= degree()) { return; }
				multIdx[changingIndex]++;
			}
		}
	}
	
	
	inline std::vector<size_t> get_step_sizes(const Tensor::DimensionTuple& _dimensions) {
		std::vector<size_t> stepSizes(_dimensions.size());
		if(!_dimensions.empty()) {
			stepSizes.back() = 1;
			for(size_t i = stepSizes.size(); i > 1; --i) {
				stepSizes[i-2] = stepSizes[i-1]*_dimensions[i-1];
			}
		}
		return stepSizes;
	}
	
	void Tensor::offset_add(const Tensor& _other, const std::vector<size_t>& _offsets) {
		IF_CHECK(
			REQUIRE(degree() == _other.degree(), "Degrees of the this and the given tensor must coincide. Here " << degree() << " vs " << _other.degree());
			REQUIRE(degree() == _offsets.size(), "Degrees of the this tensor and number of given offsets must coincide. Here " << degree() << " vs " << _offsets.size());
			for(size_t d = 0; d < degree(); ++d) {
				REQUIRE(dimensions[d] >= _offsets[d]+_other.dimensions[d], "Invalid offset/tensor dimension for dimension " << d << " because this dimension is " << dimensions[d] << " but other tensor dimension + offset is " << _offsets[d]+_other.dimensions[d]);
			}
		)
		
		if(_other.is_dense()) {
			use_dense_representation();
			
			const std::vector<size_t> stepSizes = get_step_sizes(dimensions);
		
			// Calculate the actual offset
			size_t offset = 0;
			for(size_t d = 0; d < degree(); ++d) {
				offset += _offsets[d]*stepSizes[d];
			}
			
			const size_t blockSize = _other.dimensions.back();
			const value_t* inPosition = _other.get_unsanitized_dense_data();
			value_t* outPosition = get_dense_data() + offset;
			
			misc::add_scaled(outPosition, _other.factor, inPosition, blockSize);
			
			for(size_t i = 1; i < _other.size/blockSize; ++i) {
				size_t index = degree()-2;
				size_t multStep = _other.dimensions[index];
				inPosition += blockSize;
				outPosition += stepSizes[index];
				while(i%multStep == 0) {
					outPosition -= _other.dimensions[index]*stepSizes[index]; // "reset" current index to 0
					--index;							// Advance to next index
					outPosition += stepSizes[index];	// increase next index
					multStep *= _other.dimensions[index];		// next stepSize
				}
				
				misc::add_scaled(outPosition, _other.factor, inPosition, blockSize);
			}
		} else {
			const size_t offset = multiIndex_to_position(_offsets, dimensions);
			if(is_dense()) {
				value_t* const dataPtr = get_dense_data();
				for(const auto& entry : _other.get_unsanitized_sparse_data()) {
					const size_t newPos = multiIndex_to_position(position_to_multiIndex(entry.first, _other.dimensions), dimensions) + offset;
					dataPtr[newPos] += _other.factor*entry.second;
				}
			} else {
				std::map<size_t, value_t>& data = get_sparse_data(); 
				for(const auto& entry : _other.get_unsanitized_sparse_data()) {
					const size_t newPos = multiIndex_to_position(position_to_multiIndex(entry.first, _other.dimensions), dimensions) + offset;
					data[newPos] += _other.factor*entry.second;
				}
			}
		}
	}
	
	
	void Tensor::use_dense_representation() {
		if(is_sparse()) {
			denseData.reset(new value_t[size], internal::array_deleter_vt);
			misc::set_zero(denseData.get(), size);
			for(const auto& entry : *sparseData) {
				denseData.get()[entry.first] = factor*entry.second;
			}
			
			factor = 1.0;
			sparseData.reset();
			representation = Representation::Dense;
		}
	}
	
	
	void Tensor::use_dense_representation_if_desirable() {
		if (is_sparse() && sparsity() * sparsityFactor >= size) {
			use_dense_representation();
		}
	}
	
	
	void Tensor::use_sparse_representation(const value_t _eps) {
		if(is_dense()) {
			sparseData.reset(new std::map<size_t, value_t>());
			for(size_t i = 0; i < size; ++i) {
				if(std::abs(factor*denseData.get()[i]) >= _eps) {
					sparseData->emplace_hint(sparseData->end(), i, factor*denseData.get()[i]);
				}
			}
			
			factor = 1.0;
			denseData.reset();
			representation = Representation::Sparse;
		}
	}

	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	std::string Tensor::to_string() const {
		if (degree() == 0) { return xerus::misc::to_string(operator[](0)); }
		
		std::string result;
		for (size_t i = 0; i < size; ++i) {
			result += xerus::misc::to_string(operator[](i)) + " ";
			if ((i+1) % (size / dimensions[0]) == 0) {
				result += '\n';
			} else if (degree() > 1 && (i+1) % (size / dimensions[0] / dimensions[1]) == 0) {
				result += '\t';
			} else if (degree() > 2 && (i+1) % (size / dimensions[0] / dimensions[1] / dimensions[2]) == 0) {
				result += "/ ";
			}
		}
		return result;
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal Helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	template<int sign>
	void Tensor::plus_minus_equal(Tensor& _me, const Tensor& _other) {
		REQUIRE(_me.dimensions == _other.dimensions, "In Tensor sum the dimensions must coincide. Got " << _me.dimensions << " and " << _other.dimensions);
		
		if(_me.is_dense()) {
			_me.ensure_own_data_and_apply_factor();
			if(_other.is_dense()) {
				misc::add_scaled(_me.denseData.get(), sign*_other.factor, _other.denseData.get(), _me.size);
			} else {
				add_sparse_to_full(_me.denseData, sign*_other.factor, _other.sparseData);
			}
		} else {
			if(_other.is_dense()) {
				_me.denseData.reset(new value_t[_me.size], internal::array_deleter_vt);
				misc::copy_scaled(_me.denseData.get(), sign*_other.factor, _other.denseData.get(), _me.size);
				add_sparse_to_full(_me.denseData, _me.factor, _me.sparseData);
				_me.factor = 1.0;
				_me.representation = Representation::Dense;
				_me.sparseData.reset();
			} else {
				_me.ensure_own_data_and_apply_factor();
				add_sparse_to_sparse(_me.sparseData, sign*_other.factor, _other.sparseData);
				_me.use_dense_representation_if_desirable();
			}
		}
	}
	
	
	void Tensor::add_sparse_to_full(const std::shared_ptr<value_t>& _denseData, const value_t _factor, const std::shared_ptr<const std::map<size_t, value_t>>& _sparseData) {
		for(const auto& entry : *_sparseData) {
			_denseData.get()[entry.first] += _factor*entry.second;
		}
	}
	
	
	void Tensor::add_sparse_to_sparse(const std::shared_ptr<std::map<size_t, value_t>>& _sum, const value_t _factor, const std::shared_ptr<const std::map<size_t, value_t>>& _summand) {
		for(const auto& entry : *_summand) {
			std::pair<std::map<size_t, value_t>::iterator, bool> result = _sum->emplace(entry.first, _factor*entry.second);
			if(!result.second) {
				result.first->second += _factor*entry.second;
			}
		}
	}
	
	
	size_t Tensor::multiIndex_to_position(const MultiIndex& _multiIndex, const DimensionTuple& _dimensions) {
		REQUIRE(_multiIndex.size() == _dimensions.size(), "MultiIndex has wrong degree. Given " << _multiIndex.size() << ",  expected " << _dimensions.size());
		
		size_t finalIndex = 0;
		for(size_t i = 0; i < _multiIndex.size(); ++i) {
			REQUIRE(_multiIndex[i] < _dimensions[i], "Index "<< i <<" out of bounds " << _multiIndex[i] << " >=! " << _dimensions[i]);
			finalIndex *= _dimensions[i];
			finalIndex += _multiIndex[i];
		}
		
		return finalIndex;
	}
	
	Tensor::MultiIndex Tensor::position_to_multiIndex(size_t _position, const DimensionTuple& _dimensions) {
		REQUIRE(_position < misc::product(_dimensions), "Invalid position " << _position << " given. Max size is : " << misc::product(_dimensions));
		MultiIndex index(_dimensions.size());
		
		for(size_t i = 0; i < _dimensions.size(); ++i) {
			const size_t k = _dimensions.size() - 1 - i;
			index[k] = _position%_dimensions[k];
			_position /= _dimensions[k];
		}
		
		return index;
	}
	
	
	void Tensor::ensure_own_data() {
		if(is_dense()) {
			if(!denseData.unique()) {
				value_t* const oldDataPtr = denseData.get();
				denseData.reset(new value_t[size], internal::array_deleter_vt);
				misc::copy(denseData.get(), oldDataPtr, size);
			}
		} else {
			if(!sparseData.unique()) {
				sparseData.reset(new std::map<size_t, value_t>(*sparseData));
			}
		}
	}
	
	
	void Tensor::ensure_own_data_no_copy() {
		if(is_dense()) {
			if(!denseData.unique()) {
				denseData.reset(new value_t[size], internal::array_deleter_vt);
			}
		} else {
			if(!sparseData.unique()) {
				sparseData.reset(new std::map<size_t, value_t>());
			}
		}
	}
	
	
	void Tensor::apply_factor() {
		if(has_factor()) {
			if(is_dense()) {
				if(denseData.unique()) {
					misc::scale(denseData.get(), factor, size);
				} else {
					value_t* const oldDataPtr = denseData.get();
					denseData.reset(new value_t[size], internal::array_deleter_vt);
					misc::copy_scaled(denseData.get(), factor, oldDataPtr, size);
				}
			} else {
				if(!sparseData.unique()) {
					sparseData.reset(new std::map<size_t, value_t>(*sparseData));
				}
				
				for(std::pair<const size_t, value_t>& entry : *sparseData) {
					entry.second *= factor;
				}
			}
			
			factor = 1.0;
		}
	}
	
	void Tensor::ensure_own_data_and_apply_factor() {
		if(has_factor()) {
			apply_factor(); 
		} else {
			ensure_own_data();
		}
	}
	
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	Tensor operator+(Tensor _lhs, const Tensor& _rhs) {
		_lhs += _rhs;
		return _lhs;
	}
	
	
	Tensor operator-(Tensor _lhs, const Tensor& _rhs) {
		_lhs -= _rhs;
		return _lhs;
	}
	
	
	Tensor operator*(const value_t _factor, Tensor _tensor) {
		_tensor *= _factor;
		return _tensor;
	}
	
	
	Tensor operator*(Tensor _tensor, const value_t _factor) {
		_tensor *= _factor;
		return _tensor;
	}
	
	
	Tensor operator/(Tensor _tensor, const value_t _divisor) {
		_tensor /= _divisor;
		return _tensor;
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - External functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	void contract(Tensor& _result, const Tensor& _lhs, const bool _lhsTrans, const Tensor& _rhs, const bool _rhsTrans, const size_t _numModes) {
		REQUIRE(_numModes <= _lhs.degree() && _numModes <= _rhs.degree(), "Cannot contract more indices than both tensors have. we have: " 
			<< _lhs.degree() << " and " << _rhs.degree() << " but want to contract: " << _numModes);
		
		const size_t lhsRemainOrder = _lhs.degree() - _numModes;
		const size_t lhsRemainStart = _lhsTrans ? _numModes : 0;
		const size_t lhsContractStart = _lhsTrans ? 0 : lhsRemainOrder;
		const size_t lhsRemainEnd = lhsRemainStart + lhsRemainOrder;
		
		const size_t rhsRemainOrder = _rhs.degree() - _numModes;
		const size_t rhsRemainStart = _rhsTrans ? 0 : _numModes;
		IF_CHECK(const size_t rhsContractStart = _rhsTrans ? rhsRemainOrder : 0;)
		const size_t rhsRemainEnd = rhsRemainStart + rhsRemainOrder;
		
		REQUIRE(std::equal(_lhs.dimensions.begin() + lhsContractStart, _lhs.dimensions.begin() + lhsContractStart + _numModes, _rhs.dimensions.begin() + rhsContractStart), 
				"Dimensions of the be contracted indices do not coincide. " <<_lhs.dimensions << " ("<<_lhsTrans<<") and " << _rhs.dimensions << " ("<<_rhsTrans<<") with " << _numModes);
		
		const size_t leftDim = misc::product(_lhs.dimensions, lhsRemainStart, lhsRemainEnd);
		const size_t midDim = misc::product(_lhs.dimensions, lhsContractStart, lhsContractStart+_numModes);
		const size_t rightDim = misc::product(_rhs.dimensions, rhsRemainStart, rhsRemainEnd);
		
		
		const size_t finalSize = leftDim*rightDim;
		const size_t sparsityExpectation = std::min(std::min(leftDim*_rhs.sparsity(), rightDim*_lhs.sparsity()), size_t(double(finalSize)*(1.0 - pow(1.0 - double(_lhs.sparsity())*double(_rhs.sparsity())/(double(_lhs.size)*double(_rhs.size)), double(midDim)))));
		// TODO allow Sparse*sparse --> Full
		const bool sparseResult = (_lhs.is_sparse() && _rhs.is_sparse()) || (finalSize > 64 && Tensor::sparsityFactor*sparsityExpectation < finalSize*2) ;
		const Tensor::Representation resultRepresentation = sparseResult ? Tensor::Representation::Sparse : Tensor::Representation::Dense;
		
		
		// Check whether _result has to be reset and prevent deletion of _lhs or _rhs due to being the same object as _result.
		std::unique_ptr<Tensor> tmpResult;
		Tensor* usedResult;
		if( &_result == &_lhs || &_result == &_rhs
			|| _result.representation != resultRepresentation
			|| _result.degree() != lhsRemainOrder + rhsRemainOrder
			|| _result.size != leftDim*rightDim
			|| !std::equal(_lhs.dimensions.begin() + lhsRemainStart, _lhs.dimensions.begin() + lhsRemainEnd, _result.dimensions.begin())
			|| !std::equal(_rhs.dimensions.begin() + rhsRemainStart, _rhs.dimensions.begin() + rhsRemainEnd, _result.dimensions.begin() + (lhsRemainEnd-lhsRemainStart))
		) {
			Tensor::DimensionTuple resultDim;
			resultDim.reserve(lhsRemainOrder + rhsRemainOrder);
			resultDim.insert(resultDim.end(), _lhs.dimensions.begin() + lhsRemainStart, _lhs.dimensions.begin() + lhsRemainEnd);
			resultDim.insert(resultDim.end(), _rhs.dimensions.begin() + rhsRemainStart, _rhs.dimensions.begin() + rhsRemainEnd);
			
			if(&_result == &_lhs || &_result == &_rhs) {
				tmpResult.reset(new Tensor(std::move(resultDim), resultRepresentation, Tensor::Initialisation::None));
				usedResult = tmpResult.get();
			} else {
				_result.reset(std::move(resultDim), resultRepresentation, Tensor::Initialisation::None);
				usedResult = &_result;
			}
		} else {
			usedResult = &_result;
		}
		
		
		if(!_lhs.is_sparse() && !_rhs.is_sparse()) { // Full * Full => Full
			blasWrapper::matrix_matrix_product(usedResult->override_dense_data(), leftDim, rightDim, _lhs.factor*_rhs.factor, 
											_lhs.get_unsanitized_dense_data(), _lhsTrans, midDim, 
											_rhs.get_unsanitized_dense_data(), _rhsTrans);
			
		} else if(_lhs.is_sparse() && _rhs.is_sparse() ) { // Sparse * Sparse => Sparse 
			internal::CholmodSparse::matrix_matrix_product(usedResult->override_sparse_data(), leftDim, rightDim, _lhs.factor*_rhs.factor, 
								_lhs.get_unsanitized_sparse_data(), _lhsTrans, midDim,
								_rhs.get_unsanitized_sparse_data(), _rhsTrans);
		} else {
			if(sparseResult) {
				if(_lhs.is_sparse() && !_rhs.is_sparse() && usedResult->is_sparse()) { // Sparse * Full => Sparse
					matrix_matrix_product(usedResult->override_sparse_data(), leftDim, rightDim, _lhs.factor*_rhs.factor, 
										_lhs.get_unsanitized_sparse_data(), _lhsTrans, midDim, 
										_rhs.get_unsanitized_dense_data(), _rhsTrans);
					
				} else { // Full * Sparse => Sparse
					matrix_matrix_product(usedResult->override_sparse_data(), leftDim, rightDim, _lhs.factor*_rhs.factor, 
										_lhs.get_unsanitized_dense_data(), _lhsTrans, midDim, 
										_rhs.get_unsanitized_sparse_data(), _rhsTrans);
				}
			} else {
				if(_lhs.is_sparse() && !_rhs.is_sparse()) { // Sparse * Full => Full
					matrix_matrix_product(usedResult->override_dense_data(), leftDim, rightDim, _lhs.factor*_rhs.factor, 
										_lhs.get_unsanitized_sparse_data(), _lhsTrans, midDim, 
										_rhs.get_unsanitized_dense_data(), _rhsTrans);
					
				} else { // Full * Sparse => Full
					matrix_matrix_product(usedResult->override_dense_data(), leftDim, rightDim, _lhs.factor*_rhs.factor, 
										_lhs.get_unsanitized_dense_data(), _lhsTrans, midDim, 
										_rhs.get_unsanitized_sparse_data(), _rhsTrans);
					
				}
			}
		}
		
		if (sparseResult) {
			usedResult->use_dense_representation_if_desirable();
		}
		
		if(tmpResult) {
			_result = std::move(*tmpResult);
		}
	}
	
	Tensor contract(const Tensor& _lhs, const bool _lhsTrans, const Tensor& _rhs, const bool _rhsTrans, const size_t _numModes) {
		Tensor result;
		contract(result, _lhs, _lhsTrans, _rhs, _rhsTrans, _numModes);
		return result;
	}

	
	XERUS_force_inline std::tuple<size_t, size_t, size_t> calculate_factorization_sizes(const Tensor& _input, const size_t _splitPos) {
		REQUIRE(_splitPos <= _input.degree(), "Split position must be in range.");
		
		const size_t lhsSize = misc::product(_input.dimensions, 0, _splitPos);
		const size_t rhsSize = misc::product(_input.dimensions, _splitPos, _input.degree());
		const size_t rank = std::min(lhsSize, rhsSize);
		
		return std::make_tuple(lhsSize, rhsSize, rank);
	}
	
	XERUS_force_inline void prepare_factorization_output(Tensor& _lhs, Tensor& _rhs, const Tensor& _input, const size_t _splitPos, const size_t _rank, Tensor::Representation _rep) {
		_lhs.factor = 1.0;
		_rhs.factor = 1.0;
		
		if (_lhs.representation != _rep
			|| _lhs.degree() != _splitPos+1
			|| _lhs.dimensions.back() != _rank 
			|| !std::equal(_input.dimensions.begin(), _input.dimensions.begin() + _splitPos, _lhs.dimensions.begin())) 
		{
			Tensor::DimensionTuple newDimU;
			newDimU.insert(newDimU.end(), _input.dimensions.begin(), _input.dimensions.begin() + _splitPos);
			newDimU.push_back(_rank);
			_lhs.reset(std::move(newDimU), _rep, Tensor::Initialisation::None);
		}
		
		if (_rhs.representation != _rep
			|| _rhs.degree() != _input.degree()-_splitPos+1 
			|| _rank != _rhs.dimensions.front() 
			|| !std::equal(_input.dimensions.begin() + _splitPos, _input.dimensions.end(), _rhs.dimensions.begin()+1)) 
		{
			Tensor::DimensionTuple newDimVt;
			newDimVt.push_back(_rank);
			newDimVt.insert(newDimVt.end(), _input.dimensions.begin() + _splitPos, _input.dimensions.end());
			_rhs.reset(std::move(newDimVt), _rep, Tensor::Initialisation::None);
		}
	}
	
	XERUS_force_inline void set_factorization_output(Tensor& _lhs, std::unique_ptr<value_t[]>&& _lhsData, Tensor& _rhs, 
										   std::unique_ptr<value_t[]>&& _rhsData, const Tensor& _input, const size_t _splitPos, const size_t _rank) {
		Tensor::DimensionTuple newDim;
		newDim.insert(newDim.end(), _input.dimensions.begin(), _input.dimensions.begin() + _splitPos);
		newDim.push_back(_rank);
		_lhs.reset(std::move(newDim), std::move(_lhsData));

		newDim.clear();
		newDim.push_back(_rank);
		newDim.insert(newDim.end(), _input.dimensions.begin() + _splitPos, _input.dimensions.end());
		_rhs.reset(std::move(newDim), std::move(_rhsData));
	}
	
	XERUS_force_inline void set_factorization_output(Tensor& _lhs, std::map<size_t, value_t>&& _lhsData, Tensor& _rhs, 
										   std::map<size_t, value_t>&& _rhsData, const Tensor& _input, const size_t _splitPos, const size_t _rank) {
		Tensor::DimensionTuple newDim;
		newDim.insert(newDim.end(), _input.dimensions.begin(), _input.dimensions.begin() + _splitPos);
		newDim.push_back(_rank);
		_lhs.reset(std::move(newDim), std::move(_lhsData));

		newDim.clear();
		newDim.push_back(_rank);
		newDim.insert(newDim.end(), _input.dimensions.begin() + _splitPos, _input.dimensions.end());
		_rhs.reset(std::move(newDim), std::move(_rhsData));
	}
	
	value_t calculate_svd(Tensor& _U, Tensor& _S, Tensor& _Vt, Tensor _input, const size_t _splitPos, const size_t _maxRank, const value_t _eps) {
		REQUIRE(0 <= _eps && _eps < 1, "Epsilon must be fullfill 0 <= _eps < 1.");
		
		size_t lhsSize, rhsSize, rank;
		std::tie(lhsSize, rhsSize, rank) = calculate_factorization_sizes(_input, _splitPos);
		
		std::unique_ptr<value_t[]> tmpS(new value_t[rank]);
		
		// sparse SVD becomes inefficient when the matrix is not sparse enough
		// sparse SVD is about equally fast to dense SVD when there are about N = 1.55*(min(m,n)+(max-min)/5) entries set
		// will calculate with 2 instead of 1.55 to make sure that we will certainly be slower with sparse
		// (note that the algorithm is quadratic in the sparsity)
		size_t min = std::min(lhsSize, rhsSize);
		size_t max = std::max(lhsSize, rhsSize);
		if (_input.is_sparse() && _input.sparsity() > 2*(min+(max-min)/5)) {
			_input.use_dense_representation();
		}
		
		// Calculate the actual SVD
		if(_input.is_sparse()) {
			// calculate QC in both directions
			calculate_qc(_U, _input, _input, _splitPos);
			calculate_cq(_input, _Vt, _input, 1);
			_input.use_dense_representation(); // in the very unlikely case that is it not already dense
			
			// then calculate SVD only of remaining (hopefully small) core
			Tensor UPrime, VPrime;
			std::tie(lhsSize, rhsSize, rank) = calculate_factorization_sizes(_input, 1);
			prepare_factorization_output(UPrime, VPrime, _input, 1, rank, Tensor::Representation::Dense);
			blasWrapper::svd(UPrime.override_dense_data(), tmpS.get(), VPrime.override_dense_data(), _input.get_unsanitized_dense_data(), lhsSize, rhsSize);
			
			// contract U*UPrime and VPrime*V to obtain SVD (UU', S, V'V) from orthogonal U and V as wel as the SVD (U', S, V')
			contract(_U, _U, UPrime, 1);
			contract(_Vt, VPrime, _Vt, 1);
		} else {
			prepare_factorization_output(_U, _Vt, _input, _splitPos, rank, Tensor::Representation::Dense);
			blasWrapper::svd(_U.override_dense_data(), tmpS.get(), _Vt.override_dense_data(), _input.get_unsanitized_dense_data(), lhsSize, rhsSize);
		}
		
        size_t full_rank = rank;

		// Account for hard threshold
		if(_maxRank != 0) {
			rank = std::min(rank, _maxRank);
		}
		
		// Find rank due to the Epsilon (NOTE the scaling factor can be ignored, as it does not change the ratios).
		for(size_t j = 1; j < rank; ++j) {
			if (tmpS[j] <= _eps*tmpS[0]) {
				rank = j;
				break;
			}
		}
		
		// Create tensor from diagonal values
		_S.reset(Tensor::DimensionTuple(2, rank), Tensor::Representation::Sparse);
		for(size_t i = 0; i < rank; ++i) {
			_S[i*rank+i] = std::abs(_input.factor)*tmpS[i];
		}
		
		// Account for a negative factor
		if(_input.factor < 0.0) {
			_Vt *= -1;
		}
		
		_U.resize_mode(_U.degree()-1, rank);
		_Vt.resize_mode(0, rank);

        if (rank < full_rank) { return std::abs(_input.factor)*tmpS[rank]; }
        else { return 0; }
	}
	
	
	void calculate_qr(Tensor& _Q, Tensor& _R, Tensor _input, const size_t _splitPos) {
		size_t lhsSize, rhsSize, rank;
		std::tie(lhsSize, rhsSize, rank) = calculate_factorization_sizes(_input, _splitPos);
		if (_input.is_sparse()) {
			std::map<size_t, double> qdata, rdata;
			std::tie(qdata, rdata, rank) = internal::CholmodSparse::qc(_input.get_unsanitized_sparse_data(), false, lhsSize, rhsSize, true);
			INTERNAL_CHECK(rank == std::min(lhsSize, rhsSize), "IE, sparse qr reduced rank");
			set_factorization_output(_Q, std::move(qdata), _R, std::move(rdata), _input, _splitPos, rank);
			_Q.use_dense_representation_if_desirable();
			_R.use_dense_representation_if_desirable();
		} else {
			prepare_factorization_output(_Q, _R, _input, _splitPos, rank, _input.representation);
			blasWrapper::qr(_Q.override_dense_data(), _R.override_dense_data(), _input.get_unsanitized_dense_data(), lhsSize, rhsSize);
		}
		
		_R.factor = _input.factor;
	}
	
	
	void calculate_rq(Tensor& _R, Tensor& _Q, Tensor _input, const size_t _splitPos) {
		size_t lhsSize, rhsSize, rank;
		std::tie(lhsSize, rhsSize, rank) = calculate_factorization_sizes(_input, _splitPos);
		prepare_factorization_output(_R, _Q, _input, _splitPos, rank, Tensor::Representation::Dense);
		
		if(_input.is_sparse()) {
			LOG_ONCE(warning, "Sparse RQ not yet implemented. falling back to the dense variant"); // TODO
			_input.use_dense_representation();
			blasWrapper::rq(_R.override_dense_data(), _Q.override_dense_data(), _input.get_unsanitized_dense_data(), lhsSize, rhsSize);
		} else {
			blasWrapper::rq(_R.override_dense_data(), _Q.override_dense_data(), _input.get_unsanitized_dense_data(), lhsSize, rhsSize);
		}
		
		_R.factor = _input.factor;
	}
	
	
	void calculate_qc(Tensor& _Q, Tensor& _C, Tensor _input, const size_t _splitPos) {
		size_t lhsSize, rhsSize, rank;
		std::tie(lhsSize, rhsSize, rank) = calculate_factorization_sizes(_input, _splitPos);
		
		if(_input.is_sparse()) {
			std::map<size_t, double> qdata, cdata;
			std::tie(qdata, cdata, rank) = internal::CholmodSparse::qc(_input.get_unsanitized_sparse_data(), false, lhsSize, rhsSize, false);
			set_factorization_output(_Q, std::move(qdata), _C, std::move(cdata), _input, _splitPos, rank);
			_Q.use_dense_representation_if_desirable();
			_C.use_dense_representation_if_desirable();
		} else {
			std::unique_ptr<double[]> QData, CData;
			std::tie(QData, CData, rank) = blasWrapper::qc(_input.get_unsanitized_dense_data(), lhsSize, rhsSize);
			set_factorization_output(_Q, std::move(QData), _C, std::move(CData), _input, _splitPos, rank);
		}
		
		_C.factor = _input.factor;
	}
	
	
	void calculate_cq(Tensor& _C, Tensor& _Q, Tensor _input, const size_t _splitPos) {
		size_t lhsSize, rhsSize, rank;
		std::tie(lhsSize, rhsSize, rank) = calculate_factorization_sizes(_input, _splitPos);
		
		if(_input.is_sparse()) {
			std::map<size_t, double> qdata, cdata;
			std::tie(cdata, qdata, rank) = internal::CholmodSparse::cq(_input.get_unsanitized_sparse_data(), false, rhsSize, lhsSize, false);
			set_factorization_output(_C, std::move(cdata), _Q, std::move(qdata), _input, _splitPos, rank);
			_Q.use_dense_representation_if_desirable();
			_C.use_dense_representation_if_desirable();
		} else {
			std::unique_ptr<double[]> CData, QData;
			std::tie(CData, QData, rank) = blasWrapper::cq(_input.get_unsanitized_dense_data(), lhsSize, rhsSize);
			set_factorization_output(_C, std::move(CData), _Q, std::move(QData), _input, _splitPos, rank);
		}
		
		_C.factor = _input.factor;
	}
	
	
	void pseudo_inverse(Tensor& _inverse, const Tensor& _input, const size_t _splitPos) {
		Tensor U, S, Vt;
		calculate_svd(U, S, Vt, _input, _splitPos, 0, EPSILON);
		S.modify_diagonal_entries([](value_t& _a){ _a = 1/_a;});
		contract(_inverse, Vt, true, S, false, 1);
		contract(_inverse, _inverse, false, U, true, 1);
	}
	
	Tensor pseudo_inverse(const Tensor& _input, const size_t _splitPos) {
		Tensor result;
		pseudo_inverse(result, _input, _splitPos);
		return result;
	}
	
	
	void solve_least_squares(Tensor& _X, const Tensor& _A, const Tensor& _B, const size_t _extraDegree) {
		REQUIRE(&_X != &_B && &_X != &_A, "Not supportet yet");
		const size_t degM = _B.degree() - _extraDegree;
		const size_t degN = _A.degree() - degM;
		
		REQUIRE(_A.degree() == degM+degN, "Inconsistent dimensions.");
		REQUIRE(_B.degree() == degM+_extraDegree, "Inconsistent dimensions.");
		
		// Make sure X has right dimensions
		if(	_X.degree() != degN + _extraDegree
			|| !std::equal(_X.dimensions.begin(), _X.dimensions.begin() + degN, _A.dimensions.begin() + degM)
			|| !std::equal(_X.dimensions.begin()+ degN, _X.dimensions.end(), _B.dimensions.begin() + degM))
		{
			Tensor::DimensionTuple newDimX;
			newDimX.insert(newDimX.end(), _A.dimensions.begin()+degM, _A.dimensions.end());
			newDimX.insert(newDimX.end(), _B.dimensions.begin()+degM, _B.dimensions.end());
			_X.reset(std::move(newDimX), Tensor::Representation::Dense, Tensor::Initialisation::None);
		}
		
		
		// Calculate multDimensions
		const size_t m = misc::product(_A.dimensions, 0, degM);
		const size_t n = misc::product(_A.dimensions, degM, degM+degN);
		const size_t p = misc::product(_B.dimensions, degM, degM+_extraDegree);
		
		if (_A.is_sparse()) {
			REQUIRE( p == 1, "Matrix least squares not supported in sparse");
			if (_B.is_sparse()) {
				internal::CholmodSparse::solve_sparse_rhs(
					_X.override_sparse_data(), 
					n, 
					_A.get_unsanitized_sparse_data(), 
					false, 
					_B.get_unsanitized_sparse_data(), 
					m);
			} else {
				internal::CholmodSparse::solve_dense_rhs(
					_X.override_dense_data(), 
					n, 
					_A.get_unsanitized_sparse_data(), 
					false,
					_B.get_unsanitized_dense_data(), 
					m);
			}
			
		} else { // Dense A
            if(_B.is_dense()) {
                blasWrapper::solve_least_squares(
                    _X.override_dense_data(), 
                    _A.get_unsanitized_dense_data(), m, n, 
                    _B.get_unsanitized_dense_data(), p);
            } else {
                LOG_ONCE(warning, "Sparse RHS not yet implemented (casting to dense)"); //TODO
                
                Tensor Bcpy(_B);
                Bcpy.factor = 1.0;
                Bcpy.use_dense_representation();
                
                blasWrapper::solve_least_squares(
                    _X.override_dense_data(), 
                    _A.get_unsanitized_dense_data(), m, n, 
                    Bcpy.get_unsanitized_dense_data(), p);
            }
		}
		
		// Propagate the constant factor
		_X.factor = _B.factor / _A.factor;
	}
	
	
	
	void solve(Tensor& _X, const Tensor& _A, const Tensor& _B, const size_t _extraDegree) {
		if (_A.is_sparse()) { // TODO
			solve_least_squares(_X, _A, _B, _extraDegree);
			return;
		}
		REQUIRE(&_X != &_B && &_X != &_A, "Not supportet yet");
		const size_t degM = _B.degree() - _extraDegree;
		const size_t degN = _A.degree() - degM;
		
		REQUIRE(_A.degree() == degM+degN, "Inconsistent dimensions.");
		REQUIRE(_B.degree() == degM+_extraDegree, "Inconsistent dimensions.");
		
		// Make sure X has right dimensions
		if(	_X.degree() != degN + _extraDegree
			|| !std::equal(_X.dimensions.begin(), _X.dimensions.begin() + degN, _A.dimensions.begin() + degM)
			|| !std::equal(_X.dimensions.begin()+ degN, _X.dimensions.end(), _B.dimensions.begin() + degM))
		{
			Tensor::DimensionTuple newDimX;
			newDimX.insert(newDimX.end(), _A.dimensions.begin()+degM, _A.dimensions.end());
			newDimX.insert(newDimX.end(), _B.dimensions.begin()+degM, _B.dimensions.end());
			_X.reset(std::move(newDimX), Tensor::Representation::Dense, Tensor::Initialisation::None);
		}
		
		// Calculate multDimensions
		const size_t m = misc::product(_A.dimensions, 0, degM);
		const size_t n = misc::product(_A.dimensions, degM, degM+degN);
		const size_t p = misc::product(_B.dimensions, degM, degM+_extraDegree);
		
		
		// Note that A isdense here
		if(_B.is_dense()) {
			blasWrapper::solve(
				_X.override_dense_data(), 
				_A.get_unsanitized_dense_data(), m, n, 
				_B.get_unsanitized_dense_data(), p);
		} else {
			LOG_ONCE(warning, "Sparse RHS not yet implemented (casting to dense)"); //TODO
			
			Tensor Bcpy(_B);
			Bcpy.factor = 1.0;
			Bcpy.use_dense_representation();
			
			blasWrapper::solve(
				_X.override_dense_data(), 
				_A.get_unsanitized_dense_data(), m, n, 
				Bcpy.get_unsanitized_dense_data(), p);
		}
		
		// Propagate the constant factor
		_X.factor = _B.factor / _A.factor;
	}
	
	double get_smallest_eigenvalue(Tensor& _X, const Tensor& _A) {
			REQUIRE(_A.is_dense(), "for now only dense is implemented"); //TODO implement sparse
			REQUIRE(&_X != &_A, "Not supportet yet");
			REQUIRE(_A.degree() % 2 == 0, "The tensor A needs to be an operator, i.e. has even degree");

			const size_t degN = _A.degree() / 2;

			// Calculate multDimensions
			const size_t m = misc::product(_A.dimensions, 0, degN);
			const size_t n = misc::product(_A.dimensions, degN, 2*degN);
			REQUIRE(m == n, "Inconsistent dimensions.");

			// Make sure X has right dimensions
			if(	_X.degree() != degN
				|| !std::equal(_X.dimensions.begin(), _X.dimensions.begin() + degN, _A.dimensions.begin() + degN))
			{
				Tensor::DimensionTuple newDimX;
				newDimX.insert(newDimX.end(), _A.dimensions.begin()+degN, _A.dimensions.end());
				_X.reset(std::move(newDimX), Tensor::Representation::Dense, Tensor::Initialisation::None);
			}

			std::unique_ptr<double[]> re(new double[n]);      // real eigenvalues
			std::unique_ptr<double[]> im(new double[n]);      // imaginary eigenvalues
			std::unique_ptr<double[]> rev(new double[n * n]); // right eigenvectors

			// Note that A is dense here
			blasWrapper::solve_ev(
				rev.get(), // right eigenvectors
				re.get(),  // real eigenvalues
				im.get(),  // imaginary eigenvalues
				_A.get_unsanitized_dense_data(),  n);

			//get eigenvalue with minimal real part
			double ev = re[0];
			size_t idx = 0;
			for (size_t i = 1; i < n; ++i) {
				if (re[i] < ev){
					ev = re[i];
					idx = i;
				}
			}

			//get eigenvector oof smallest eigenvalue
			auto tmpX = _X.override_dense_data();
			for (size_t i = 0; i < n; ++i)
				tmpX[i] = rev[idx + i * n];

			return ev;
		}
	
	
	Tensor entrywise_product(const Tensor &_A, const Tensor &_B) {
		REQUIRE(_A.dimensions == _B.dimensions, "Entrywise product ill-defined for non-equal dimensions.");
		if(_A.is_dense() && _B.is_dense()) {
			Tensor result(_A);
			
			result *= _B.factor;
			value_t* const dataPtrA = result.get_dense_data();
			const value_t* const dataPtrB = _B.get_unsanitized_dense_data();
			for(size_t i = 0; i < result.size; ++i) {
				dataPtrA[i] *= dataPtrB[i];
			}
			return result;
		}
		if(_A.is_sparse()) {
			Tensor result(_A);
			
			for(std::pair<const size_t, value_t>& entry : result.get_sparse_data()) {
				entry.second *= _B[entry.first];
			}
			return result;
		} 
		// _B.is_sparse()
		Tensor result(_B);
		
		for(std::pair<const size_t, value_t>& entry : result.get_sparse_data()) {
			entry.second *= _A[entry.first];
		}
		return result;
	}
	
	bool approx_equal(const xerus::Tensor& _a, const xerus::Tensor& _b, const xerus::value_t _eps) {
		REQUIRE(_a.dimensions == _b.dimensions, 
				"The dimensions of the compared tensors don't match: " << _a.dimensions <<" vs. " << _b.dimensions << " and " << _a.size << " vs. " << _b.size);
		
		return frob_norm(_a - _b) <= _eps*(_a.frob_norm() + _b.frob_norm())/2.0;
	}
	
	bool approx_entrywise_equal(const Tensor& _a, const Tensor& _b, const value_t _eps) {
		REQUIRE(_a.dimensions == _b.dimensions, 
				"The dimensions of the compared tensors don't match: " << _a.dimensions <<" vs. " << _b.dimensions << " and " << _a.size << " vs. " << _b.size);
		
		if (_a.is_sparse() && _b.is_sparse()) { // Special treatment if both are sparse, because better asyptotic is possible.
			for(const auto& entry : _a.get_unsanitized_sparse_data()) {
				if(!misc::approx_equal(_a.factor*entry.second, _b[entry.first], _eps)) { return false; }
			}
			
			for(const auto& entry : _b.get_unsanitized_sparse_data()) {
				if(!misc::approx_equal(_a[entry.first], _b.factor*entry.second, _eps)) { return false; }
			}
		} else {
			for(size_t i=0; i < _a.size; ++i) {
				if(!misc::approx_equal(_a[i], _b[i], _eps)) { return false; }
			}
		}
		return true;
	}
	
	bool approx_entrywise_equal(const xerus::Tensor& _tensor, const std::vector<value_t>& _values, const xerus::value_t _eps) {
		if(_tensor.size != _values.size()) { return false; }
		for(size_t i = 0; i < _tensor.size; ++i) {
			if(!misc::approx_equal(_tensor[i], _values[i], _eps)) { return false; }
		}
		return true;
	}
	
	std::ostream& operator<<(std::ostream& _out, const Tensor& _tensor) {
		_out << _tensor.to_string();
		return _out;
	}
	
	namespace misc {
		
		
		void stream_writer(std::ostream &_stream, const Tensor &_obj, const FileFormat _format) {
			if(_format == FileFormat::TSV) {
				_stream << std::setprecision(std::numeric_limits<value_t>::digits10 + 1);
			}
			
			// storage version number
			write_to_stream<size_t>(_stream, 1, _format);
			
			write_to_stream(_stream, _obj.dimensions, _format);
			
			if (_obj.representation == Tensor::Representation::Dense) {
				write_to_stream<size_t>(_stream, 1, _format);
				for (size_t i = 0; i < _obj.size; ++i) {
					write_to_stream<value_t>(_stream, _obj[i], _format);
				}
			} else {
				write_to_stream<size_t>(_stream, 2, _format);
				write_to_stream<size_t>(_stream, _obj.get_unsanitized_sparse_data().size(), _format);
				for (const auto &d : _obj.get_unsanitized_sparse_data()) {
					write_to_stream<size_t>(_stream, d.first, _format);
					write_to_stream<value_t>(_stream, _obj.factor*d.second, _format);
				}
			}
		}
		
		
		void stream_reader(std::istream& _stream, Tensor &_obj, const FileFormat _format) {
			IF_CHECK(size_t ver = )read_from_stream<size_t>(_stream, _format);
			REQUIRE(ver == 1, "Unknown stream version to open (" << ver << ")");
			
			// Load dimensions
			std::vector<size_t> dims;
			read_from_stream(_stream, dims, _format);
			
			// Load representation
			const size_t rep = read_from_stream<size_t>(_stream, _format);
			
			// Load data
			if (rep == 1) { // Dense
				_obj.reset(std::move(dims), Tensor::Representation::Dense, Tensor::Initialisation::None);
				if(_format == FileFormat::TSV) {
					for (size_t i = 0; i < _obj.size; ++i) {
						_stream >> (_obj.get_unsanitized_dense_data()[i]);
					}
				} else {
					_stream.read(reinterpret_cast<char*>(_obj.get_unsanitized_dense_data()), std::streamoff(_obj.size*sizeof(value_t)));
				}
				REQUIRE(_stream, "Unexpected end of stream in reading dense Tensor.");
			} else { // Sparse 
				REQUIRE(rep == 2, "Unknown tensor representation " << rep << " in stream");
				_obj.reset(std::move(dims), Tensor::Representation::Sparse);
				
				const uint64 num = read_from_stream<size_t>(_stream, _format);
				REQUIRE(num < std::numeric_limits<size_t>::max(), "The stored Tensor is to large to be loaded using 32 Bit xerus.");
				
				for (size_t i = 0; i < num; ++i) {
					REQUIRE(_stream, "Unexpected end of stream in reading sparse Tensor.");
					// NOTE inline function calls can be called in any order by the compiler, so we have to cache the results to ensure correct order
					uint64 pos = read_from_stream<size_t>(_stream, _format);
					value_t val = read_from_stream<value_t>(_stream, _format);
					_obj.get_unsanitized_sparse_data().emplace(pos, val);
				}
				REQUIRE(_stream, "Unexpected end of stream in reading dense Tensor.");
			}
		}
	} // namespace misc
} // namespace xerus
