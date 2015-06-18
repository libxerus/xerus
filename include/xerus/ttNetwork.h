// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2015 Benjamin Huber and Sebastian Wolf. 
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
 * @brief Header file for the TTNetwork class (and thus TTTensor and TTOperator).
 */

#pragma once

#include "contractionHeuristic.h"
#include "fullTensor.h"
#include "index.h"
#include "misc/missingFunctions.h"
#include "misc/check.h"

namespace xerus {

	/**
	 * @brief Specialized TensorNetwork class used to represent TTTensor and TToperators.
	 * @details TTTensors correspond to isOperator=FALSE and TTOperators correspond to isOperator=FALSE.
	 */
    template<bool isOperator>
    class TTNetwork final : public TensorNetwork {
	public:
		///@brief The number of external links in each node, i.e. one for TTTensors and two for TTOperators.
		static constexpr const size_t N=isOperator?2:1;
		
		/// @brief Flag indicating whether the TTNetwork is cannonicalized.
		bool cannonicalized;
		
		/**
		 * @brief The position of the core.
		 * @details If cannonicalized is TRUE, corePosition gives the position of the core tensor. All components
		 * with smaller index are then left-orthogonalized and all components with larger index right-orthogonalized.
		 */
		size_t corePosition;		
		
        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		/** 
		 * @brief Constructs an order zero TTNetwork.
		 * @details This is a TensorNetwork with no entries. The global factor (which is the only entry of the tensor)
		 * is set to zero.
		 */
        explicit TTNetwork();
        
        /** 
		 * @brief Constructs an zero initialized TTNetwork with the given degree and ranks all equal to one.
		 * @details Naturally for TTOperators the degree must be even.
		 */
		explicit TTNetwork(const size_t _degree);
        
        /** 
        * @brief Constructs a TTNetwork from the given FullTensor.
        * @details  The higher order SVD algorithm is used to decompose the given Tensor into the TT format.
        * @param _tensor The Tensor to decompose.
        * @param _eps the accuracy to be used in the decomposition.
        * @param _maxRank the maximal allowed rank (applies to all positions).
        */
        explicit TTNetwork(const Tensor& _tensor, const double _eps=EPSILON, const size_t _maxRank=std::numeric_limits<size_t>::max());
        
        /** 
        * @brief Constructs a TTNetwork from the given FullTensor.
        * @details  The higher order SVD algorithm is used to decompose the given Tensor into the TT format.
        * @param _tensor The Tensor to decompose.
        * @param _eps the accuracy to be used in the decomposition.
        * @param _maxRanks maximal ranks to be used
        */
        explicit TTNetwork(const Tensor& _tensor, const double _eps, const std::vector<size_t>& _maxRanks);
        
		///@brief Copy constructor for TTNetworks.
        implicit TTNetwork(const TTNetwork & _cpy);
        
		///@brief Move constructor for TTNetworks.
        implicit TTNetwork(      TTNetwork&& _mov);
        
        /** 
		* @brief Transforms a given TensorNetwork to a TTNetwork.
		* @details This is not yet implemented different from casting to FullTensor and then using a HOSVD.
		* @param _network The network to transform.
		* @param _eps the accuracy to be used in the transformation.
		*/
		explicit TTNetwork(const TensorNetwork &_network, double _eps=EPSILON);
        
		/// Random constructs a TTNetwork with the given dimensions and ranks. The entries of the componend tensors are sampled independendly using the provided random generator and distribution.
        template<class generator, class distribution>
        static TTNetwork construct_random(const std::vector<size_t>& _dimensions, const std::vector<size_t> &_ranks, generator& _rnd, distribution& _dist) {
            REQUIRE(_ranks.size() == _dimensions.size()/N-1,"Non-matching amount of ranks given to TTNetwork::construct_random");
            #ifndef DISABLE_RUNTIME_CHECKS_
                for (const size_t d : _dimensions) {
                    REQUIRE(d > 0, "Trying to construct random TTTensor with dimension 0 is illegal.");
                }
                for (const size_t d : _ranks) {
                    REQUIRE(d > 0, "Trying to construct random TTTensor with rank 0 is illegal.");
                }
            #endif
            
            TTNetwork result(_dimensions.size());
			const size_t numComponents = _dimensions.size()/N;
			size_t maxDim1 = 1;
			size_t maxDim2 = misc::product(_dimensions);
			
			for(size_t i = 0; i < numComponents; ++i) {
				size_t oldMaxDim = std::min(maxDim1, maxDim2);
				maxDim1 *= _dimensions[i] * (isOperator? _dimensions[numComponents+i] : 1);
				maxDim2 /= _dimensions[i] * (isOperator? _dimensions[numComponents+i] : 1);
				size_t maxDim = std::min(maxDim1, maxDim2);
				if(isOperator) {
					result.set_component(i, FullTensor::construct_random({i==0?1:std::min(oldMaxDim, _ranks[i-1]), _dimensions[i], _dimensions[numComponents+i], i==numComponents-1?1:std::min(maxDim, _ranks[i])}, _rnd, _dist));
				} else {
					result.set_component(i, FullTensor::construct_random({i==0?1:std::min(oldMaxDim, _ranks[i-1]), _dimensions[i], i==numComponents-1?1:std::min(maxDim, _ranks[i])}, _rnd, _dist));
				}
			}
            result.cannonicalize_left();
            REQUIRE(result.is_valid_tt(), "Internal Error.");
            return result;
        }
        
		/// Random constructs a TTNetwork with the given dimensions and ranks. The entries of the componend tensors are sampled independendly using the provided random generator and distribution.
        template<class generator, class distribution>
        static TTNetwork construct_random(const std::vector<size_t>& _dimensions, size_t _rank, generator& _rnd, distribution& _dist) {
            return construct_random(_dimensions, std::vector<size_t>(_dimensions.size()/N-1, _rank), _rnd, _dist);
        }
        
        /// Construct a TTOperator with the given dimensions representing the identity. (Only applicable for TTOperators, i.e. not for TTtensors).
        template<bool B = isOperator, typename std::enable_if<B, int>::type = 0>
        static TTNetwork construct_identity(const std::vector<size_t>& _dimensions);
        
		
        TTNetwork& operator=(const TTNetwork& _other) = default;
        
        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
    protected:
        void round_train(const std::vector<size_t>& _maxRanks, const double _eps);

		static void construct_train_from_full(TensorNetwork& _out, const FullTensor& _A, const double _eps);
        
        static void contract_stack(const IndexedTensorWritable<TensorNetwork> &_me);
            
        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
    public:
        /** 
		* @brief Tests whether the network resembles that of a TTTensor and checks consistency with the underlying tensor objects.
		* @details Note that this will NOT check for orthogonality of cannonicalized TTNetworks.
		* @return TRUE if the check is passed, otherwise an exception is thrown and the function does not return.
		*/
		bool is_valid_tt() const;
		
		/** 
		* @brief Computes the dyadic product of @a _lhs and @a _rhs. 
		* @details This function is currently needed to keep the resulting network in the TTNetwork class.
		* Apart from that the result is the same as result(i^d1, j^d2) = _lhs(i^d1)*_rhs(j^d2).
		* @returns the dyadic product as a TTNetwork.
		*/
		static TTNetwork dyadic_product(const TTNetwork &_lhs, const TTNetwork &_rhs);
		
		/** 
		 * @brief Computes the dyadic product of all given TTNetworks. 
		 * @details This is nothing but the repeated application of dyadic_product() for the given TTNetworks.
		 * @returns the dyadic product as a TTNetwork.
		 */
		static TTNetwork dyadic_product(const std::vector<std::reference_wrapper<TTNetwork>> &_tensors);
		
		/**
		 * @brief Calculates the componentwise product of two tensors given in the TT format.
		 * @details Usually the resulting rank = rank(A)*rank(B). Retains the core position of @a _A
		 */
		static TTNetwork entrywise_product(const TTNetwork &_A, const TTNetwork &_B);
		
		/** 
		* @brief Complete access to a specific component of the TT decomposition.
		* @note This function will not update rank and external dimension informations if it is used to set a component.
		* @details This function gives complete access to the components, only intended for internal use.
		* @param _idx index of the component to access.
		* @returns a reference to the requested component.
		*/
		Tensor& component(const size_t _idx);
		
		/** 
		* @brief Read access to a specific component of the TT decomposition.
		* @details This function should be used to access the components, instead of direct access via
		* nodes[...], because the implementation does not store the first component in nodes[0] but rather as
		* nodes[1] etc. nodes[0] is an order one node with dimension one only used to allow the first component
		* to be an order three tensor.
		* @param _idx index of the component to access.
		* @returns a const reference to the requested component.
		*/
		const Tensor &get_component(const size_t _idx) const;
		
		/** 
		* @brief Sets a specific component of the TT decomposition.
		* @details This function also takes care of adjusting the corresponding link dimensions and external dimensions
		* if needed. However this might still leave the TTNetwork in an invalid if the rank is changed. In this case it
		* is the callers responsibility to also update the other component tensors consistently to account for that rank
		* change.
		* @param _idx index of the component to set.
		* @param _T Tensor to use as the new component tensor.
		*/
		void set_component(const size_t _idx, const Tensor &_T);
		
		/** 
		* @brief Sets a specific component of the TT decomposition.
		* @details This function also takes care of adjusting the corresponding link dimensions and external dimensions
		* if needed. However this might still leave the TTNetwork in an invalid if the rank is changed. In this case it
		* is the callers responsibility to also update the other component tensors consistently to account for that rank
		* change.
		* @param _idx index of the component to set.
		* @param _T Tensor to use as the new component tensor.
		*/
		void set_component(size_t _idx, std::unique_ptr<Tensor> &&_T);
		
		/** 
		* @brief Splits the TTNetwork into two parts by removing the node.
		* @param _position index of the component to be removed, thereby also defining the position 
		*of the split.
		* @return a std::pair containing the two remaining parts as TensorNetworks.
		*/
        std::pair<TensorNetwork, TensorNetwork> chop(const size_t _position) const;
        
		/** 
		 * @brief Reduce all ranks up to a given accuracy.
		 * @param _eps the accuracy to use for truncation in the individual SVDs.
		 */
        void round(value_t _eps);

		/** 
		 * @brief Reduce all ranks to the given number.
		 * @param _maxRank maximal allowed rank. All current ranks that are larger than this are reduced by truncation.
		 */
        void round(size_t _maxRank);

		/** 
		 * @brief Reduce all ranks to the given numbers.
		 * @param _maxRanks maximal allowed ranks. All current ranks that are larger than the given ones are reduced by truncation.
		 */
        void round(const std::vector<size_t> &_maxRanks);
        
		/** 
		 * @brief Reduce all ranks to the given number.
		 * @param _maxRank maximal allowed rank. All current ranks that are larger than this are reduced by truncation.
		 */
        void round(int _maxRank);
		
		/** 
		 * @brief Applies the soft threshholding operation to all ranks.
		 * @param _tau the soft threshholding parameter to be applied. I.e. all singular values are reduced to max(0, Lambda_ui - _tau).
		 */
        void soft_threshold(const double _tau);
		
		/** 
		 * @brief Applies soft threshholding operations to all ranks.
		 * @param _taus the soft threshholding parameters to be applied. I.e. all singular values of the j-th matrification are reduced to max(0, Lambda_ui - _tau[j]).
		 */
        void soft_threshold(const std::vector<double>& _taus);
		
		

		/** 
		 * @brief Gets the ranks of the TTNetwork.
		 * @return A vector containing the current ranks.
		 */
        std::vector<size_t> ranks() const;
        
		// TODO describtion
        size_t rank(size_t _i) const;
        
		/** 
		 * @brief Calculates the storage requirement of the current representation.
		 * @return The datasize in sizeof(value_t).
		 */
        size_t datasize() const;
        
		/** 
		 * @brief Move the core to a new position.
		 * @details The core is moved to @a _position and the nodes between the old and the new position are orthogonalized
		 * accordingly. If the TTNetwork is not yet cannonicalized it will be with @a _position as new corePosition.
		 * @param _position the new core position.
		 * @param _keepRank by default a rank revealing QR decomposition is used to move the core and the ranks are reduced
		 * accordingly. If @a _keepRank is set the rank is not reduced, this is need e.g. in the ALS.
		 */
		void move_core(size_t _position, bool _keepRank=false);
		
		/**
		 * @brief stores @a _pos as the current core position without verifying of ensuring that this is the case
		 * @details this is particularly useful after constructing an own TT tensor with set_component calls
		 * as these will assume that all orthogonalities are destroyed
		 */
		void assume_core_position(size_t _pos) {
			REQUIRE(_pos < degree() / N, "invalid core position");
			corePosition = _pos;
			cannonicalized = true;
		}
		
		/** 
		 * @brief Move the core to the left.
		 * @details Basically calls move_core() with _position = 0
		 */
        void cannonicalize_left();
        
		/** 
		 * @brief Move the core to the left.
		 * @details Basically calls move_core() with _position = degree()-1
		 */
        void cannonicalize_right();
            
		/** 
		 * @brief Transpose the TTOperator
		 * @details Swaps all external indices to create the transposed operator.
		 */
        template<bool B = isOperator, typename std::enable_if<B, int>::type = 0>
        void transpose() {
            Index i,r,l,j;
            for (size_t n=0; n < degree()/N; ++n) {
				std::unique_ptr<Tensor> At(nodes[n+1].tensorObject->construct_new());
                (*At)(l,j,i,r) = get_component(n)(l,i,j,r);
				set_component(n, std::move(At));
            }
        }
        
        
        virtual TensorNetwork* get_copy() const override;
        
        virtual value_t frob_norm() const override;
        
        virtual bool is_in_expected_format() const override;
        
        /*- - - - - - - - - - - - - - - - - - - - - - - - - -  Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - */
        /** 
		 * @brief Adds a given TTNetwork to this one.
		 * @details To be well-defined it is required that the dimensions of this and @a _other coincide. 
		 * The rank of the result are in general the entrywise sum of the ranks of this and @a _other.
		 * @param _other the TTNetwork to add.
		 * @return reference to this TTNetwork.
		 */
		TTNetwork& operator+=(const TTNetwork& _other);
		
		/** 
		 * @brief Calculates the entrywise sum of this TTNetwork and @a _other.
		 * @details To be well-defined it is required that the dimensions of this and @a _other coincide. 
		 * The rank of the result are in general the entrywise sum of the ranks of this and @a _other.
		 * @param _other the second summand.
		 * @return the sum.
		 */
        TTNetwork  operator+(const TTNetwork& _other) const;
        
		/** 
		 * @brief Subtracts the @a _other TTNetwork entrywise from this one.
		 * @details To be well-defined it is required that the dimensions of this and @a _other coincide. 
		 * The rank of the result are in general the entrywise sum of the ranks of this and @a _other.
		 * @param _other the Tensor to be subtracted to this one.
		 * @return a reference to this TTNetwork.
		 */
        TTNetwork& operator-=(const TTNetwork& _other);
		
		/** 
		 * @brief Calculates the entrywise difference between this TTNetwork and @a _other.
		 * @details To be well-defined it is required that the dimensions of this and @a _other coincide. 
		 * The rank of the result are in general the entrywise sum of the ranks of this and @a _other.
		 * @param _other the subtrahend,
		 * @return the difference.
		 */
        TTNetwork  operator-(const TTNetwork& _other) const;
        
		
        virtual void operator*=(const value_t _factor) override;
		
		
		/** 
		 * @brief Calculates the entrywise multiplication of this TensorNetwork with a constant @a _factor.
		 * @details Internally this only results in a change in the global factor.
		 * @param _factor the factor,
		 * @return the resulting scaled TensorNetwork.
		 */
        TTNetwork  operator*(const value_t _factor) const;
        
        virtual void operator/=(const value_t _divisor) override;
		
		/** 
		 * @brief Calculates the entrywise divison of this TensorNetwork by a constant @a _divisor.
		 * @details Internally this only results in a change in the global factor.
		 * @param _divisor the divisor,
		 * @return the resulting scaled TensorNetwork.
		 */
        TTNetwork  operator/(const value_t _div) const;
        
        
        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
		static bool specialized_contraction_f(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other);
		static bool specialized_sum_f(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other);
        virtual bool specialized_contraction(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) const override {
			return specialized_contraction_f(_out, _me, _other);
		}
        virtual bool specialized_sum(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) const override {
			return specialized_sum_f(_out, _me, _other);
		}
        virtual void specialized_evaluation(const IndexedTensorWritable<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) override;
        
    };

    template<bool isOperator>
    static _inline_ TTNetwork<isOperator> operator*(const value_t _lhs, const TTNetwork<isOperator>& _rhs) { return _rhs*_lhs; }


    typedef TTNetwork<false> TTTensor;
    typedef TTNetwork<true> TTOperator;

    namespace internal {
        template<bool isOperator>
        /// Internal class used to represent stacks of (possibly multiply) applications of TTOperators to either a TTTensor or TTOperator.
        class TTStack final : public TensorNetwork {
        public:
			bool cannonicalization_required;
			size_t futureCorePosition;
			
			explicit TTStack(bool _canno, size_t _corePos=0) : cannonicalization_required(_canno), futureCorePosition(_corePos) {};
			
			
			virtual void operator*=(const value_t _factor) override {
				REQUIRE(nodes.size() > 0, "There must not be a TTNetwork without any node");
				
				if(cannonicalization_required) {
					*nodes[futureCorePosition+1].tensorObject *= _factor;
				} else if(degree() > 0) {
					*nodes[1].tensorObject *= _factor;
				} else {
					*nodes[0].tensorObject *= _factor;
				}
			}
		
			virtual void operator/=(const value_t _divisor) override {
				operator*=(1/_divisor);
			}
			
		
            /*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
            virtual void specialized_evaluation(const IndexedTensorWritable<TensorNetwork> &_me _unused_ , const IndexedTensorReadOnly<TensorNetwork> &_other _unused_) override {
                LOG(fatal, "TTStack not supported as a storing type");
            }
            virtual bool specialized_contraction(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) const override {
				return TTNetwork<isOperator>::specialized_contraction_f(_out, _me, _other);
			}
			virtual bool specialized_sum(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) const override {
				return TTNetwork<isOperator>::specialized_sum_f(_out, _me, _other);
			}
            
            virtual TensorNetwork* get_copy() const override {
                LOG(fatal, "Forbidden");
                return nullptr;
            }
            
            virtual value_t frob_norm() const override {
                Index i;
                TTNetwork<isOperator> tmp(this->degree());
                tmp(i&0) = (*this)(i&0);
                return tmp.frob_norm();
            }
        };
    }
}
