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
 * @brief Implementation of the HTStack classes.
 */

#include <xerus/htStack.h>
#include <xerus/basic.h>
#include <xerus/misc/check.h>
#include <xerus/misc/internal.h>

#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/htNetwork.h>
 

namespace xerus {
	namespace internal {
		template<bool isOperator>
		HTStack<isOperator>::HTStack(const bool _canno, const size_t _corePos) : cannonicalization_required(_canno), futureCorePosition(_corePos) {}
		
		
		template<bool isOperator>
		TensorNetwork* HTStack<isOperator>::get_copy() const {
			return new HTStack(*this);
		}
		
		
		template<bool isOperator>
		HTStack<isOperator>::operator HTNetwork<isOperator>() {
		require_valid_network();

		if(degree() == 0) {
			std::set<size_t> toContract;
			for(size_t i = 0; i < nodes.size(); ++i) {
				toContract.insert(i);
				contract(toContract);
				return HTNetwork<isOperator>(*nodes[0].tensorObject);
			}
		}
		const size_t numOfLeaves = degree()/N;
		const size_t numOfFullLeaves = numOfLeaves == 1 ? 2 : static_cast<size_t>(0.5+std::pow(2,std::ceil(std::log2(static_cast<double>(numOfLeaves)))));
		const size_t numIntComp = numOfFullLeaves - 1;
		const size_t numComponents = numIntComp + numOfLeaves;
		const size_t numNodes = numOfFullLeaves + numIntComp + 1;
		const size_t stackSize = nodes.size()/numNodes;

		INTERNAL_CHECK(nodes.size()%numNodes == 0, "IE");

		// Contract the stack to a HTNetwork node structure.
		std::set<size_t> toContract;
		for (size_t currentNode = 0; currentNode < numNodes; ++currentNode) {
			toContract.clear();
			for (size_t i = 0; i < stackSize; i++) {
				toContract.insert(currentNode+i*numNodes);
			}
			contract(toContract);
		}
		// Reshuffle the nodes to be in the correct order after contraction the nodes will have one of the ids: node, node+numNodes, node+2*numNodes,... (as those were part of the contraction) so modulus gives the correct wanted id.
		reshuffle_nodes([numNodes](const size_t _i){return _i%(numNodes);});

		INTERNAL_CHECK(nodes.size() == numNodes, "Internal Error.");

		// Reset to new external links
		for(size_t i = 0; i < numOfLeaves; ++i) {
			externalLinks[i].other = i+numIntComp;
			externalLinks[i].indexPosition = 1;
		}
		if(isOperator) {
			for(size_t i = 0; i < numOfLeaves; ++i) {
				externalLinks[numOfLeaves+i].other = i+numIntComp;
				externalLinks[numOfLeaves+i].indexPosition = 2;
			}
		}


		// Fix the virtual node
		nodes[numNodes-1].tensorObject->reinterpret_dimensions({1});
		nodes[numNodes-1].neighbors.resize(1);
		nodes[numNodes-1].neighbors.front().other = 0;
		nodes[numNodes-1].neighbors.front().indexPosition = 0;

		// Fix all internal components
		std::vector<size_t> shuffle_int(3*stackSize);
		for (size_t i = 0; i < numIntComp; ++i) {
			size_t parentCount = 0;
			size_t parentDim = 1, leftChildDim = 1, rightChildDim = 1;
			size_t fullDim = 1;
			for(size_t k = 0; k < 3*stackSize; ++k) {
				INTERNAL_CHECK(!nodes[i].erased, "IE");
				const TensorNetwork::Link& link = nodes[i].neighbors[k];
				fullDim *= link.dimension;
				//link is pointing to parent
				if((i > 0 && link.other == (i-1)/2) || (i == 0 && link.other == numNodes - 1)) {
					shuffle_int[k] = parentCount++;
					parentDim *= link.dimension;
				}
				// link is pointing to left child
				else if (link.other == 2 * i + 1){
					// We want the order of the next node (so the next one can keep its order)
					size_t otherPos = 0;
					for(const TensorNetwork::Link& otherLink : nodes[2 * i + 1].neighbors) {
						if(otherLink.other == i) {
							if(otherLink.indexPosition == k) {
								break;
							} else {
								otherPos++;
							}
						}
					}
					shuffle_int[k] = stackSize+otherPos;
					leftChildDim *= link.dimension;
				}
				// link is pointing to right child
				else if (link.other == 2 * i + 2){
					size_t otherPos = 0;
					for(const TensorNetwork::Link& otherLink : nodes[2 * i + 2].neighbors) {
						if(otherLink.other == i) {
							if(otherLink.indexPosition == k) {
								break;
							} else {
								otherPos++;
							}
						}
					}
					shuffle_int[k] = 2*stackSize+otherPos;
					rightChildDim *= link.dimension;
				}
				else
					INTERNAL_CHECK(false, "Internal Error, something wrong with the links of the TTN");
			}
			INTERNAL_CHECK(fullDim == nodes[i].tensorObject->size, "Uhh");
			INTERNAL_CHECK(parentCount == stackSize, "IE");
			xerus::reshuffle(*nodes[i].tensorObject, *nodes[i].tensorObject, shuffle_int);
			nodes[i].tensorObject->reinterpret_dimensions({parentDim, leftChildDim, rightChildDim});
			nodes[i].neighbors.clear();
			nodes[i].neighbors.emplace_back(i == 0 ? numNodes - 1 : (i-1) / 2, i==0 ? 0 : (i%2 == 1 ? 1:2), parentDim, false);
			nodes[i].neighbors.emplace_back(2*i+1, 0, leftChildDim, false);
			nodes[i].neighbors.emplace_back(2*i+2, 0, rightChildDim, false);
		}
		// Fix all leaves
		std::vector<size_t> shuffle_leaves(N + stackSize);
		for (size_t i = numIntComp; i < numNodes - 1; ++i) {
			size_t parentCount = 0, parentDim = 1, fullDim = 1;
			//Dummy components
			if(i >= numComponents){
				nodes[i].tensorObject->reinterpret_dimensions({1});
				nodes[i].neighbors.resize(1);
				nodes[i].neighbors.front().other = (i-1)/2;
				nodes[i].neighbors.front().indexPosition = (i-1)%2 + 1;
			} else {
			for(size_t k = 0; k < N + stackSize; ++k) {
				INTERNAL_CHECK(!nodes[i].erased, "IE");
				const TensorNetwork::Link& link = nodes[i].neighbors[k];
				fullDim *= link.dimension;
				if(link.external) {
					if(link.indexPosition < numOfLeaves) {
						shuffle_leaves[k] = stackSize;
					} else {
						INTERNAL_CHECK(isOperator, "IE " << link.indexPosition << " vs " << numOfLeaves << " vs " << degree());
						shuffle_leaves[k] = stackSize+1;
					}
				} else {
						//link is pointing to parent
						if(( link.other == (i-1)/2)) {
							shuffle_leaves[k] = parentCount++;
							parentDim *= link.dimension;
						}
						else
							INTERNAL_CHECK(false, "Internal Error, something wrong with the links of the TTN");
					}
				}
				INTERNAL_CHECK(fullDim == nodes[i].tensorObject->size, "Uhh");
				INTERNAL_CHECK(parentCount == stackSize, "IE");
				xerus::reshuffle(*nodes[i].tensorObject, *nodes[i].tensorObject, shuffle_leaves);
				if(isOperator) {
					nodes[i].tensorObject->reinterpret_dimensions({parentDim, dimensions[i - numIntComp], dimensions[i - numIntComp+numOfLeaves]});
				} else {
					nodes[i].tensorObject->reinterpret_dimensions({parentDim, dimensions[i - numIntComp]});
				}

				nodes[i].neighbors.clear();
				nodes[i].neighbors.emplace_back( (i-1) / 2, i%2 == 1 ? 1:2, parentDim, false);
				nodes[i].neighbors.emplace_back(-1, i - numIntComp , dimensions[i - numIntComp], true);
				if(isOperator) { nodes[i].neighbors.emplace_back(-1, numOfLeaves+i - numIntComp, dimensions[numOfLeaves+i - numIntComp], true); }
			}
		}
		// Create actual HTNetwork
		HTNetwork<isOperator> result;
		static_cast<TensorNetwork&>(result) = static_cast<TensorNetwork&>(*this);
		if(cannonicalization_required) {
			result.canonicalized = false;
			result.move_core(futureCorePosition);
		} else {
			result.canonicalized = true;
			result.corePosition = futureCorePosition;
		}
		result.require_correct_format();

		return result;
	}
		
		template<bool isOperator>
		void HTStack<isOperator>::operator*=(const value_t _factor) {
			INTERNAL_CHECK(!nodes.empty(), "There must not be a HTNetwork without any node");
			
			if(cannonicalization_required) {
				*nodes[futureCorePosition+1].tensorObject *= _factor;
			} else if(degree() > 0) {
				*nodes[1].tensorObject *= _factor;
			} else {
				*nodes[0].tensorObject *= _factor;
			}
		}
		
		
		template<bool isOperator>
		void HTStack<isOperator>::operator/=(const value_t _divisor) {
			operator*=(1/_divisor);
		}
		
		
		// TODO get rid of this function and use TTN cast instead
		template<bool isOperator>
		void HTStack<isOperator>::contract_stack(IndexedTensorWritable<TensorNetwork>&& _me) {
			_me.tensorObject->require_valid_network();

			if(_me.tensorObject->degree() == 0) { //TODO check this
				std::set<size_t> toContract;
				for(size_t i = 0; i < _me.tensorObject->nodes.size(); ++i) {
					toContract.insert(i);
					_me.tensorObject->contract(toContract);
					_me.tensorObject->reshuffle_nodes([](const size_t _i){return 0;});
					return;
				}
			}
			const size_t numOfLeaves = _me.tensorObject->degree()/N;
			const size_t numOfFullLeaves = numOfLeaves == 1 ? 2 : static_cast<size_t>(0.5+std::pow(2,std::ceil(std::log2(static_cast<double>(numOfLeaves)))));
			const size_t numIntComp = numOfFullLeaves - 1;
			const size_t numComponents = numIntComp + numOfLeaves;
			const size_t numNodes = numOfFullLeaves + numIntComp + 1;
			const size_t stackSize = _me.tensorObject->nodes.size()/numNodes;

			INTERNAL_CHECK(_me.tensorObject->nodes.size()%numNodes == 0, "IE");

			// Contract the stack to a HTNetwork node structure.
			std::set<size_t> toContract;
			for (size_t currentNode = 0; currentNode < numNodes; ++currentNode) {
				toContract.clear();
				for (size_t i = 0; i < stackSize; i++) {
					toContract.insert(currentNode+i*numNodes);
				}
				_me.tensorObject->contract(toContract);
			}
			// Reshuffle the nodes to be in the correct order after contraction the nodes will have one of the ids: node, node+numNodes, node+2*numNodes,... (as those were part of the contraction) so modulus gives the correct wanted id.
			_me.tensorObject->reshuffle_nodes([numNodes](const size_t _i){return _i%(numNodes);});

			INTERNAL_CHECK(_me.tensorObject->nodes.size() == numNodes, "Internal Error.");

			// Reset to new external links
			for(size_t i = 0; i < numOfLeaves; ++i) {
				_me.tensorObject->externalLinks[i].other = i+numIntComp;
				_me.tensorObject->externalLinks[i].indexPosition = 1;
			}
			if(isOperator) {
				for(size_t i = 0; i < numOfLeaves; ++i) {
					_me.tensorObject->externalLinks[numOfLeaves+i].other = i+numIntComp;
					_me.tensorObject->externalLinks[numOfLeaves+i].indexPosition = 2;
				}
			}


			// Fix the virtual node
			_me.tensorObject->nodes[numNodes-1].tensorObject->reinterpret_dimensions({1});
			_me.tensorObject->nodes[numNodes-1].neighbors.resize(1);
			_me.tensorObject->nodes[numNodes-1].neighbors.front().other = 0;
			_me.tensorObject->nodes[numNodes-1].neighbors.front().indexPosition = 0;

			// Fix all internal components
			std::vector<size_t> shuffle_int(3*stackSize);
			for (size_t i = 0; i < numIntComp; ++i) {
				size_t parentCount = 0;
				size_t parentDim = 1, leftChildDim = 1, rightChildDim = 1;
				size_t fullDim = 1;
				for(size_t k = 0; k < 3*stackSize; ++k) {
					INTERNAL_CHECK(!_me.tensorObject->nodes[i].erased, "IE");
					const TensorNetwork::Link& link = _me.tensorObject->nodes[i].neighbors[k];
					fullDim *= link.dimension;
					//link is pointing to parent
					if((i > 0 && link.other == (i-1)/2) || (i == 0 && link.other == numNodes - 1)) {
						shuffle_int[k] = parentCount++;
						parentDim *= link.dimension;
					}
					// link is pointing to left child
					else if (link.other == 2 * i + 1){
						// We want the order of the next node (so the next one can keep its order)
						size_t otherPos = 0;
						for(const TensorNetwork::Link& otherLink : _me.tensorObject->nodes[2 * i + 1].neighbors) {
							if(otherLink.other == i) {
								if(otherLink.indexPosition == k) {
									break;
								} else {
									otherPos++;
								}
							}
						}
						shuffle_int[k] = stackSize+otherPos;
						leftChildDim *= link.dimension;
					}
					// link is pointing to right child
					else if (link.other == 2 * i + 2){
						size_t otherPos = 0;
						for(const TensorNetwork::Link& otherLink : _me.tensorObject->nodes[2 * i + 2].neighbors) {
							if(otherLink.other == i) {
								if(otherLink.indexPosition == k) {
									break;
								} else {
									otherPos++;
								}
							}
						}
						shuffle_int[k] = 2*stackSize+otherPos;
						rightChildDim *= link.dimension;
					}
					else
						INTERNAL_CHECK(false, "Internal Error, something wrong with the links of the TTN");
				}
				INTERNAL_CHECK(fullDim == _me.tensorObject->nodes[i].tensorObject->size, "Uhh");
				INTERNAL_CHECK(parentCount == stackSize, "IE");
				xerus::reshuffle(*_me.tensorObject->nodes[i].tensorObject, *_me.tensorObject->nodes[i].tensorObject, shuffle_int);
				_me.tensorObject->nodes[i].tensorObject->reinterpret_dimensions({parentDim, leftChildDim, rightChildDim});
				_me.tensorObject->nodes[i].neighbors.clear();
				_me.tensorObject->nodes[i].neighbors.emplace_back(i == 0 ? numNodes - 1 : (i-1) / 2, i==0 ? 0 : (i%2 == 1 ? 1:2), parentDim, false);
				_me.tensorObject->nodes[i].neighbors.emplace_back(2*i+1, 0, leftChildDim, false);
				_me.tensorObject->nodes[i].neighbors.emplace_back(2*i+2, 0, rightChildDim, false);
			}
			// Fix all leaves
			std::vector<size_t> shuffle_leaves(N + stackSize);
			for (size_t i = numIntComp; i < numNodes - 1; ++i) {
				size_t parentCount = 0, parentDim = 1, fullDim = 1;
				//Dummy components
				if(i >= numComponents){
					_me.tensorObject->nodes[i].tensorObject->reinterpret_dimensions({1});
					_me.tensorObject->nodes[i].neighbors.resize(1);
					_me.tensorObject->nodes[i].neighbors.front().other = (i-1)/2;
					_me.tensorObject->nodes[i].neighbors.front().indexPosition = (i-1)%2 + 1;
				} else {
				for(size_t k = 0; k < N + stackSize; ++k) {
					INTERNAL_CHECK(!_me.tensorObject->nodes[i].erased, "IE");
					const TensorNetwork::Link& link = _me.tensorObject->nodes[i].neighbors[k];
					fullDim *= link.dimension;
					if(link.external) {
						if(link.indexPosition < numOfLeaves) {
							shuffle_leaves[k] = stackSize;
						} else {
							INTERNAL_CHECK(isOperator, "IE " << link.indexPosition << " vs " << numOfLeaves << " vs " << _me.tensorObject->degree());
							shuffle_leaves[k] = stackSize+1;
						}
					} else {
							//link is pointing to parent
							if(( link.other == (i-1)/2)) {
								shuffle_leaves[k] = parentCount++;
								parentDim *= link.dimension;
							}
							else
								INTERNAL_CHECK(false, "Internal Error, something wrong with the links of the TTN");
						}
					}
					INTERNAL_CHECK(fullDim == _me.tensorObject->nodes[i].tensorObject->size, "Uhh");
					INTERNAL_CHECK(parentCount == stackSize, "IE");
					xerus::reshuffle(*_me.tensorObject->nodes[i].tensorObject, *_me.tensorObject->nodes[i].tensorObject, shuffle_leaves);
					if(isOperator) {
						_me.tensorObject->nodes[i].tensorObject->reinterpret_dimensions({parentDim, _me.tensorObject->dimensions[i - numIntComp], _me.tensorObject->dimensions[i - numIntComp+numOfLeaves]});
					} else {
						_me.tensorObject->nodes[i].tensorObject->reinterpret_dimensions({parentDim, _me.tensorObject->dimensions[i - numIntComp]});
					}

					_me.tensorObject->nodes[i].neighbors.clear();
					_me.tensorObject->nodes[i].neighbors.emplace_back( (i-1) / 2, i%2 == 1 ? 1:2, parentDim, false);
					_me.tensorObject->nodes[i].neighbors.emplace_back(-1, i - numIntComp , _me.tensorObject->dimensions[i - numIntComp], true);
					if(isOperator) { _me.tensorObject->nodes[i].neighbors.emplace_back(-1, numOfLeaves+i - numIntComp, _me.tensorObject->dimensions[numOfLeaves+i - numIntComp], true); }
				}
			}
			_me.tensorObject->require_valid_network();

		}
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
		template<bool isOperator>
		void HTStack<isOperator>::specialized_evaluation(IndexedTensorWritable<TensorNetwork>&&   /*_me*/, IndexedTensorReadOnly<TensorNetwork>&&  /*_other*/) {
			LOG(fatal, "HTStack not supported as a storing type");
		}
		
		
		template<bool isOperator>
		bool HTStack<isOperator>::specialized_contraction(std::unique_ptr<IndexedTensorMoveable<TensorNetwork>>& _out, IndexedTensorReadOnly<TensorNetwork>&& _me, IndexedTensorReadOnly<TensorNetwork>&& _other) const {
			return HTNetwork<isOperator>::specialized_contraction_f(_out, std::move(_me), std::move(_other));
		}
		
		
//		template<bool isOperator>
//		bool HTStack<isOperator>::specialized_sum(std::unique_ptr<IndexedTensorMoveable<TensorNetwork>>& _out, IndexedTensorReadOnly<TensorNetwork>&& _me, IndexedTensorReadOnly<TensorNetwork>&& _other) const {
//			return HTNetwork<isOperator>::specialized_sum_f(_out, std::move(_me), std::move(_other));
//		}
		
		
		template<bool isOperator>
		value_t HTStack<isOperator>::frob_norm() const {
			const Index i;
			HTNetwork<isOperator> tmp;
			tmp(i&0) = IndexedTensorMoveable<TensorNetwork>(this->get_copy(), {i&0});
			return tmp.frob_norm();
		}
		
		
		// Explicit instantiation of the two template parameters that will be implemented in the xerus library
		template class HTStack<false>;
		template class HTStack<true>;
	} // namespace internal
} // namespace xerus
