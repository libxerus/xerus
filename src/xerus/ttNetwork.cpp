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

#include <xerus/ttNetwork.h>
#include <xerus/tensorNetwork.h>
#include <xerus/basic.h>
#include <xerus/index.h>
#include <xerus/fullTensor.h>
#include <xerus/sparseTensor.h>
#include <xerus/indexedTensorList.h>
#include <xerus/indexedTensor_TN_operators.h>
#include <xerus/indexedTensor_tensor_operators.h>
#include <xerus/indexedTensor_tensor_factorisations.h>
#include <xerus/misc/blasLapackWrapper.h>

namespace xerus {
    /*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(const size_t _degree) : TensorNetwork(), cannonicalization(RIGHT) {
		REQUIRE(_degree%N==0, "illegal degree for TTOperator");
		const size_t numComponents = _degree/N;
		factor = 0.0;
		
		if (numComponents == 0) {
			return;
		}
		
		dimensions = std::vector<size_t>(_degree, 1);
		
		// externalLinks
		for (size_t i = 1; i <= numComponents; ++i) {
			externalLinks.emplace_back(i, 1, 1, false);
		}
		if (N == 2) {
			for (size_t i = 1; i <= numComponents; ++i) {
				externalLinks.emplace_back(i, 2, 1, false);
			}
		}
		
		REQUIRE(externalLinks.size() == _degree, "ie");
		
		std::vector<TensorNode::Link> neighbors;
		
		neighbors.emplace_back(1,0,1,false);
		
		nodes.emplace_back(
			std::unique_ptr<Tensor>(new FullTensor({1},[](){return 1.0;})), 
			std::move(neighbors)
		);
		for (size_t i=1; i<=numComponents; ++i) {
			neighbors.clear();
			neighbors.emplace_back(i-1, i==1?0:N+1, 1, false);
			for (size_t j=0; j< N; ++j) { 
				neighbors.emplace_back(-1, i+j*numComponents, 1, true);
			}
			neighbors.emplace_back(i+1, 0, 1, false);
			
			nodes.emplace_back(
				std::unique_ptr<Tensor>(new FullTensor(neighbors.size())), 
				std::move(neighbors)
			);
		}
		neighbors.clear();
		neighbors.emplace_back(numComponents, N+1, 1, false);
		nodes.emplace_back(
			std::unique_ptr<Tensor>(new FullTensor({1},[](){return 1.0;})), 
			std::move(neighbors)
		);
		
		REQUIRE(is_valid_tt(), "ie");
	}
	
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(const FullTensor& _full, const double _eps): TTNetwork(_full.degree()) {
		REQUIRE(_eps < 1, "_eps must be smaller than one. " << _eps << " was given.");
		REQUIRE(_full.degree()%N==0, "Number of indicis must be even for TTOperator");
		
		dimensions = _full.dimensions;
		
		if (_full.degree() == 0) { 
			factor = _full.data.get()[0];
			return; 
		}
		factor = 1.0;
		
		const size_t numComponents = degree()/N;
		
		// Needed variables
		std::unique_ptr<Tensor> nxtTensor;
		std::unique_ptr<value_t[]> currentU, currentS;
		std::shared_ptr<value_t> workingData, currentVt;
		size_t leftDim=1, remainingDim=_full.size, maxRank, newRank=1, oldRank=1;
		
		// If we want a TTOperator we need to reshuffle the indices first, otherwise we want to copy the data because Lapack wants to destroy it
		if (N==1) {
			currentVt.reset(new value_t[_full.size], internal::array_deleter_vt);
			array_copy(currentVt.get(), _full.data.get(), _full.size);
		} else {
			FullTensor tmpTensor(degree());
			std::vector<Index> presentIndices, newIndices;
			for(size_t i = 0; i < degree(); ++i) { presentIndices.emplace_back(); }
			for(size_t i = 0; i < numComponents; ++i) {
				newIndices.emplace_back(presentIndices[i]);
				newIndices.emplace_back(presentIndices[i+numComponents]);
			}
			tmpTensor(newIndices) = _full(presentIndices);
			currentVt = tmpTensor.data;
		}
		
		for(size_t position = 0; position < numComponents-1; ++position) {
			workingData = std::move(currentVt);
			oldRank = newRank;
			
			// determine dimensions of the next matrification
			leftDim = oldRank*dimensions[position-1];
			if(N == 2) { leftDim *= dimensions[position+numComponents-1]; }
			remainingDim /= dimensions[position-1];
			if(N == 2) { remainingDim /= dimensions[position+numComponents-1]; }
			maxRank = std::min(leftDim, remainingDim);
			
			// create temporary space for the results
			currentU.reset(new value_t[leftDim*maxRank]);
			currentS.reset(new value_t[maxRank]);
			currentVt.reset(new value_t[maxRank*remainingDim]);
			
			blasWrapper::svd_destructive(currentU.get(), currentS.get(), currentVt.get(), workingData.get(), leftDim, remainingDim);
			
			// determine the rank, keeping all singular values that are large enough
			newRank = maxRank;
			while (currentS[newRank-1] < _eps*currentS[0]) {
				newRank-=1;
			}
			
			// Create a FullTensor for U
			std::vector<size_t> dimensions;
			dimensions.emplace_back(oldRank);
			dimensions.emplace_back(dimensions[position-1]);
			if (N == 2) { dimensions.emplace_back(dimensions[position+numComponents-1]); }
			dimensions.emplace_back(newRank);
			if (newRank == maxRank) {
				nxtTensor.reset(new FullTensor(std::move(dimensions), std::move(currentU)) );
			} else {
				nxtTensor.reset(new FullTensor(std::move(dimensions), DONT_SET_ZERO()) );
				for (size_t i = 0; i < leftDim; ++i) {
					array_copy(static_cast<FullTensor*>(nxtTensor.get())->data.get()+i*newRank, currentU.get()+i*maxRank, newRank);
				}
			}
			
			// update component tensor to U
			set_component(position, std::move(nxtTensor));
			
			// Calclate S*Vt by scaling the rows of Vt
			for (size_t row = 0; row < newRank; ++row) {
				array_scale(currentVt.get()+row*remainingDim, currentS[row], remainingDim);
			}
		}
		
		// Create FullTensor for Vt
		if (N==1) {
			nxtTensor.reset(new FullTensor({oldRank, dimensions[degree()-1], 1}, DONT_SET_ZERO()) );
		} else {
			nxtTensor.reset(new FullTensor({oldRank, dimensions[degree()/N-1], dimensions[degree()-1], 1}, DONT_SET_ZERO()) );
		}
		array_copy(static_cast<FullTensor*>(nxtTensor.get())->data.get(), workingData.get(), oldRank*remainingDim);
		
		// set last component tensor to Vt
		set_component(numComponents-1, std::move(nxtTensor));
		
		cannonicalization = RIGHT;
		
		REQUIRE((N==1 && remainingDim == _A.dimensions.back()) || (N==2 && remainingDim == _A.dimensions[_A.degree()/2-1]*_A.dimensions[_A.degree()-1]), "Internal Error");
		REQUIRE(_out.is_in_expected_format(), "ie");
	}
	
	
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(const TTNetwork & _cpy) : TensorNetwork(_cpy), cannonicalization(_cpy.cannonicalization) { }
	
	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(      TTNetwork&& _mov) : TensorNetwork(std::move(_mov)), cannonicalization(_mov.cannonicalization) { }

	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(const TensorNetwork &_cpy, double _eps) : TensorNetwork(_cpy) {
		LOG(fatal, "Cast of arbitrary tensor network to TT not yet supported"); // TODO
	}

	template<bool isOperator>
	TTNetwork<isOperator>::TTNetwork(TensorNetwork &&_mov, double _eps) : TensorNetwork(std::move(_mov)) {
		LOG(fatal, "Cast of arbitrary tensor network to TT not yet supported"); // TODO
	}
	
	template<bool isOperator>
	TTNetwork<isOperator> TTNetwork<isOperator>::construct_identity(const std::vector<size_t>& _dimensions) {
		REQUIRE(isOperator, "tttensor identity ill-defined");
		REQUIRE(_dimensions.size()%2==0, "illegal number of dimensions for ttOperator");
		#ifndef DISABLE_RUNTIME_CHECKS_
		for (size_t d : _dimensions) {
			REQUIRE(d > 0, "trying to construct TTOperator with dimension 0");
		}
		#endif
		
		TTNetwork result(_dimensions.size());
		result.factor = 1.0;
		
		size_t numComponents = _dimensions.size()/N;
		
		std::vector<size_t> constructionVector;
		for (size_t i=0; i<numComponents; ++i) {
			constructionVector.clear();
			constructionVector.push_back(1);
			for (size_t j=0; j< N; ++j) { 
				constructionVector.push_back(_dimensions[i+j*numComponents]);
			}
			constructionVector.push_back(1);
			set_component(i, std::unique_ptr<Tensor>(new FullTensor(constructionVector, [](const std::vector<size_t> &_idx){
				if (_idx[1] == _idx[2]) {
					return 1.0;
				} else {
					return 0.0;
				}
			})));
		}
		
		REQUIRE(result.is_valid_tt(), "ie");
		result.cannonicalize_right();
		return result;
	}
	
	
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
    template<bool isOperator>
    void TTNetwork<isOperator>::round_train(TensorNetwork& _me, const std::vector<size_t>& _maxRanks, const double _eps) {
        REQUIRE(_me.degree()%N==0, "Number of indicis must be even for TTOperator");
        REQUIRE(_eps < 1, "_eps must be smaller than one. " << _eps << " was given.");
        REQUIRE(_maxRanks.size() == _me.degree()/N-1, "There must be exactly degree/N-1 maxRanks. Here " << _maxRanks.size() << " instead of " << _me.degree()/N-1 << " are given.");
        
        // If there is no or only one node in the train object we are already finished as there is no rank that can be rounded
        if(_me.degree() <= N) { return; }
        
        // Needed variables
        std::unique_ptr<value_t[]> LR, RR, M, U, S, Vt, newLeft, newRight;
        size_t leftDim, midDim, rightDim, maxLeftRank, maxRightRank, maxRank, rank;
        
        for(size_t position = 0; position < _me.degree()/N-1; ++position) {
            REQUIRE(dynamic_cast<FullTensor*>(_me.nodes[position].tensorObject.get()), "Tensor Train nodes are required to be FullTensors");
            REQUIRE(dynamic_cast<FullTensor*>(_me.nodes[position+1].tensorObject.get()), "Tensor Train nodes are required to be FullTensors");
            
            FullTensor& leftTensor = *static_cast<FullTensor*>(_me.nodes[position].tensorObject.get());
            FullTensor& rightTensor = *static_cast<FullTensor*>(_me.nodes[position+1].tensorObject.get());
            
            // Determine the dimensions of the next matrifications
            REQUIRE(leftTensor.dimensions.back() == rightTensor.dimensions.front(), "Internal Error");
            midDim   = leftTensor.dimensions.back();
            leftDim  = leftTensor.size/midDim;
            rightDim = rightTensor.size/midDim;
            maxLeftRank = std::min(leftDim, midDim);
            maxRightRank = std::min(midDim, rightDim);
            maxRank = std::min(maxLeftRank, maxRightRank);
            
            // Calculate QR and RQ decompositoins
            LR.reset(new value_t[maxLeftRank*midDim]);
            RR.reset(new value_t[midDim*maxRightRank]);
            blasWrapper::inplace_qr(leftTensor.data.get(), LR.get(), leftDim, midDim);
            blasWrapper::inplace_rq(RR.get(), rightTensor.data.get(), midDim, rightDim);
            
            // Calculate Middle Matrix M = LR*RR
            M.reset(new value_t[maxLeftRank*maxRightRank]);
            blasWrapper::matrix_matrix_product(M.get(), maxLeftRank, maxRightRank, 1.0, LR.get(), false, midDim, RR.get(), false);
            
            // Calculate SVD of U S Vt = M -- Reuse the space allocted for LR and RR and allow destruction of M
            U = std::move(LR);
            S.reset(new value_t[maxRank]);
            Vt = std::move(RR);
            blasWrapper::svd_destructive(U.get(), S.get(), Vt.get(), M.get(), maxLeftRank, maxRightRank);
            
            //Determine the Rank
            for(rank = maxRank; S[rank-1] < _eps*S[0]; --rank) { }
            rank = std::min(rank, _maxRanks[position]);
            
            // Calclate S*Vt by scaling the rows of Vt appropriatly
            for(size_t row = 0; row < rank; ++row) {
                array_scale(Vt.get()+row*maxRightRank, S[row], maxRightRank);
            }
            
            //Multiply U and (S*Vt) back to the left and to the right
            newLeft.reset(new value_t[leftDim*rank]);
            newRight.reset(new value_t[rank*rightDim]);
            blasWrapper::matrix_matrix_product(newLeft.get(), leftDim, rank, 1.0, leftTensor.data.get(), maxLeftRank, false, maxLeftRank, U.get(), maxRank, false);
            blasWrapper::matrix_matrix_product(newRight.get(), rank, rightDim, 1.0, Vt.get(), false, maxRightRank, rightTensor.data.get(), false);
            
            // Put the new Tensors to their place
            leftTensor.data.reset(newLeft.release(), internal::array_deleter_vt);
            rightTensor.data.reset(newRight.release(), internal::array_deleter_vt);
            leftTensor.dimensions.back() = rank;
            leftTensor.size = product(leftTensor.dimensions);
            rightTensor.dimensions.front() = rank;
            rightTensor.size = product(rightTensor.dimensions);
            _me.nodes[position  ].neighbors.back().dimension  = rank;
            _me.nodes[position+1].neighbors.front().dimension = rank;
        }
        REQUIRE(_me.is_in_expected_format(), "ie");
    }
    
    
    template<bool isOperator>
    void TTNetwork<isOperator>::contract_stack(const IndexedTensorWritable<TensorNetwork> &_me) {
        REQUIRE(_me.tensorObject->is_valid_network(), "cannot contract inconsistent ttStack");
        const size_t N = isOperator?2:1;
        const size_t numNodes = _me.degree()/N;
        std::set<size_t> toContract;
        for (size_t currentNode=0; currentNode < numNodes; ++currentNode) {
            toContract.clear();
            for (size_t i=currentNode; i<_me.tensorObject->nodes.size(); i+=numNodes) {
                toContract.insert(i);
            }
            _me.tensorObject->contract(toContract);
        }
        // all are contracted, reshuffle them to be in the correct order
        // after contraction the nodes will have one of the ids: node, node+numNodes, node+2*numNodes,... (as those were part of the contraction)
        // so modulus gives the correct wanted id
        _me.tensorObject->reshuffle_nodes([numNodes](size_t i){return i%(numNodes);});
        REQUIRE(_me.tensorObject->nodes.size() == numNodes, "ie");
        REQUIRE(_me.tensorObject->is_valid_network(), "something went wrong in contract_stack");
        
        // reset to new external links
        _me.tensorObject->externalLinks.clear();
        _me.tensorObject->externalLinks.emplace_back(0, 0, _me.tensorObject->dimensions[0], false);
        for(size_t i = 1; i < numNodes; ++i) {
            _me.tensorObject->externalLinks.emplace_back(i, 1, _me.tensorObject->dimensions[i], false);
        }
        if(N == 2) {
            _me.tensorObject->externalLinks.emplace_back(0, 1, _me.tensorObject->dimensions[numNodes], false);
            for(size_t i = 1; i < numNodes; ++i) {
                _me.tensorObject->externalLinks.emplace_back(i, 2, _me.tensorObject->dimensions[numNodes+i], false);
            }
        }
        
        // ensure right amount and order of links
        Index ext[N];
        size_t lastRank, externalDim[N], newRank;
        std::vector<Index> lastIndices, lastRight;
        std::vector<Index> oldIndices, newRight; // newLeft == lastRight
        std::vector<Index> newIndices;
        std::vector<size_t> newDimensions;
        for (size_t i=0; i<numNodes; ++i) {
            lastIndices = std::move(oldIndices); oldIndices.clear();
            lastRight = std::move(newRight); newRight.clear();
            lastRank = newRank; newRank=1;
            TensorNode &n = _me.tensorObject->nodes[i];
            for (TensorNode::Link &l : n.neighbors) {
                if (l.external) {
                    size_t externalNumber = 0;
                    if (N==2) {
                        externalNumber = l.indexPosition>=numNodes?1:0;
                    }
                    oldIndices.push_back(ext[externalNumber]);
                    externalDim[externalNumber] = l.dimension;
                } else if (i >= 1 && l.links(i-1)) {
                    REQUIRE(lastIndices.size() > l.indexPosition, "ie " << lastIndices.size() << " " << l.indexPosition);
                    oldIndices.push_back(lastIndices[l.indexPosition]);
                } else if (l.links(i+1)) {
                    oldIndices.emplace_back();
                    newRight.push_back(oldIndices.back());
                    newRank *= l.dimension;
                } else  {
                    LOG(fatal, "ie");
                }
            }
            newIndices = std::move(lastRight);
            newIndices.insert(newIndices.end(), ext, ext+N);
            newIndices.insert(newIndices.end(), newRight.begin(), newRight.end());
            
            (*n.tensorObject)(newIndices) = (*n.tensorObject)(oldIndices);
            
            newDimensions.clear();
            n.neighbors.clear();
            if (i>0) {
                n.neighbors.emplace_back(i-1,(i>1? N+1 : N),lastRank, false);
                newDimensions.push_back(lastRank);
            }
            for (size_t j=0; j<N; ++j) {
                REQUIRE(_me.tensorObject->dimensions[i+j*numNodes] == externalDim[j], "ie");
                n.neighbors.emplace_back(0,i+j*numNodes,externalDim[j], true);
                newDimensions.push_back(externalDim[j]);
            }
            if (i<numNodes-1) {
                n.neighbors.emplace_back(i+1,0,newRank, false);
                newDimensions.push_back(newRank);
            }
            n.tensorObject->reinterpret_dimensions(newDimensions);
        }
        
        
        REQUIRE(_me.tensorObject->is_in_expected_format(), "something went wrong in contract_stack");
    }
    
    #ifndef DISABLE_RUNTIME_CHECKS_
        template<bool isOperator>
        bool TTNetwork<isOperator>::is_valid_tt() const {
            const size_t N = isOperator?2:1;
            const size_t numNodes = degree()/N;
            REQUIRE(nodes.size() == numNodes, nodes.size() << " vs " << numNodes);
            REQUIRE(externalLinks.size() == degree(), externalLinks.size() << " vs " << degree());
            REQUIRE(std::isfinite(factor), factor);
            
            // Per external link
            for (size_t n=0; n<externalLinks.size(); ++n) {
                const TensorNode::Link &l = externalLinks[n];
                REQUIRE(l.dimension == dimensions[n], "n=" << n << " " << l.dimension << " vs " << dimensions[n]);
                REQUIRE(!l.external, "n=" << n);
                REQUIRE(l.other < numNodes, "n=" << n << " " << l.other << " vs " << numNodes);
                REQUIRE(l.indexPosition < nodes[l.other].neighbors.size(), "n=" << n << " " << l.indexPosition << " vs " << nodes[l.other].neighbors.size());
                REQUIRE(nodes[l.other].neighbors[l.indexPosition].external, "n=" << n);
                REQUIRE(nodes[l.other].neighbors[l.indexPosition].indexPosition == n, "n=" << n << " " << nodes[l.other].neighbors[l.indexPosition].indexPosition);
                REQUIRE(nodes[l.other].neighbors[l.indexPosition].dimension == l.dimension, "n=" << n << " " << nodes[l.other].neighbors[l.indexPosition].dimension << " vs " << l.dimension);
            }
            
            // Per node
            for (size_t n=0; n<numNodes; ++n) {
                const TensorNode &node = nodes[n];
                REQUIRE(!node.erased, "n=" << n);
                if (n==0) { // first node (or only node)
                    REQUIRE(node.degree() == N+(numNodes>1?1:0), "n=" << n << " " << node.degree());
                    if (node.tensorObject) {
                        REQUIRE(node.tensorObject->degree() == N+(numNodes>1?1:0), "n=" << n << " " << node.tensorObject->degree());
                    }
                    REQUIRE(node.neighbors[0].external, "n=" << n);
                    REQUIRE(node.neighbors[0].indexPosition == n, "n=" << n << " " << node.neighbors[0].indexPosition);
                    if (isOperator) {
                        REQUIRE(node.neighbors[1].external, "n=" << n);
                        REQUIRE(node.neighbors[1].indexPosition == numNodes+n, "n=" << n << " " << node.neighbors[1].indexPosition << " vs " << numNodes+n);
                    }
                } else {
                    REQUIRE(node.degree() == N+(n<numNodes-1?2:1), "n=" << n << " " << node.degree());
                    if (node.tensorObject) {
                        REQUIRE(node.tensorObject->degree() == N+(n<numNodes-1?2:1), "n=" << n << " " << node.tensorObject->degree());
                    }
                    REQUIRE(!node.neighbors[0].external, "n=" << n);
                    REQUIRE(node.neighbors[0].other == n-1, "n=" << n);
                    REQUIRE(node.neighbors[0].indexPosition == N+(n>1?1:0), "n=" << n << " " << node.neighbors[0].indexPosition);
                    REQUIRE(node.neighbors[1].external, "n=" << n);
                    REQUIRE(node.neighbors[1].indexPosition == n, "n=" << n << " " << node.neighbors[0].indexPosition);
                    if (isOperator) {
                        REQUIRE(node.neighbors[2].external, "n=" << n);
                        REQUIRE(node.neighbors[2].indexPosition == numNodes+n, "n=" << n << " " << node.neighbors[1].indexPosition << " vs " << numNodes+n);
                    }
                }
                
                if (n < numNodes-1) {
                    if (node.tensorObject) {
                        REQUIRE(!node.tensorObject->has_factor(), "n="<<n);
                    }
                    REQUIRE(!node.neighbors.back().external, "n=" << n);
                    REQUIRE(node.neighbors.back().other == n+1, "n=" << n << " " << node.neighbors.back().other);
                    REQUIRE(node.neighbors.back().indexPosition == 0, "n=" << n << " " << node.neighbors.back().indexPosition);
                    REQUIRE(!nodes[n+1].neighbors.empty(), "n=" << n);
                    REQUIRE(node.neighbors.back().dimension == nodes[n+1].neighbors[0].dimension, "n=" << n << " " << node.neighbors.back().dimension << " vs " << nodes[n+1].neighbors[0].dimension);
                    
                } 
            }
            
            return true;
        }
    #else
        template<bool isOperator>
        bool TTNetwork<isOperator>::is_valid_tt() const {
            return true;
        }
    #endif
    
    /*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
    
	template<bool isOperator>
	const Tensor &TTNetwork::get_component(size_t _idx) const {
		REQUIRE(_idx < degree()/N, "illegal index in TTNetwork::get_component");
		return *nodes[_idx].tensorObject;
	}
	
	
	template<bool isOperator>
	void TTNetwork::set_component(size_t _idx, const Tensor &_T) {
		REQUIRE(_idx < degree()/N, "illegal index in TTNetwork::set_component");
		TensorNode &currNode = nodes[_idx+1];
		REQUIRE(_T.degree() == currNode.degree(), "degree of _T does not match component tensors degree");
		currNode.tensorObject.reset(_T.get_copy());
		for (size_t i=0; i<currNode.degree(); ++i) {
			currNode.neighbors[i].dimension = currNode.tensorObject->dimensions[i];
			if (currNode.neighbors[i].external) {
				externalLinks[currNode.neighbors[i].indexPosition].dimension = currNode.tensorObject->dimensions[i];
			}
		}
	}
	
	template<bool isOperator>
	void TTNetwork::set_component(size_t _idx, std::unique_ptr<Tensor> &&_T) {
		REQUIRE(_idx < degree()/N, "illegal index in TTNetwork::set_component");
		TensorNode &currNode = nodes[_idx+1];
		REQUIRE(_T.degree() == currNode.degree(), "degree of _T does not match component tensors degree");
		currNode.tensorObject.reset(std::move(_T));
		for (size_t i=0; i<currNode.degree(); ++i) {
			currNode.neighbors[i].dimension = currNode.tensorObject->dimensions[i];
			if (currNode.neighbors[i].external) {
				externalLinks[currNode.neighbors[i].indexPosition].dimension = currNode.tensorObject->dimensions[i];
			}
		}
	}
	
    template<bool isOperator>
    TTNetwork<isOperator> TTNetwork<isOperator>::dyadic_product(const TTNetwork<isOperator> &_lhs, const TTNetwork<isOperator> &_rhs) {
        REQUIRE(_lhs.is_in_expected_format(), "");
        REQUIRE(_rhs.is_in_expected_format(), "");
        
        if (_lhs.degree() == 0) {
            TTNetwork result(_rhs);
            result.factor *= _lhs.factor;
            return result;
        }
        TTNetwork result(_lhs);
        result.factor *= _rhs.factor;
        if (_rhs.degree() == 0) {
            return result;
        }
        
        // add all nodes of rhs and fix neighbor relations
        const size_t lhsNodesSize = _lhs.nodes.size();
        const size_t rhsNodesSize = _rhs.nodes.size();
        for (const TensorNode &n : _rhs.nodes) {
            result.nodes.emplace_back(n);
            for (TensorNode::Link &l : result.nodes.back().neighbors) {
                if (l.external) {
                    if (l.indexPosition < rhsNodesSize) {
                        l.indexPosition += lhsNodesSize;
                    } else {
                        l.indexPosition += 2*lhsNodesSize;
                    }
                } else {
                    l.other += lhsNodesSize;
                }
            }
        }
        
        // add rank-1 connection between the two
        std::vector<size_t> dimensions(result.nodes[lhsNodesSize-1].tensorObject->dimensions);
        dimensions.push_back(1);
        result.nodes[lhsNodesSize-1].tensorObject->reinterpret_dimensions(dimensions);
        result.nodes[lhsNodesSize-1].neighbors.emplace_back(lhsNodesSize, 0, 1, false);
        const size_t linkPos = dimensions.size()-1;
        
        dimensions.clear();
        dimensions.push_back(1);
        dimensions.insert(dimensions.end(), result.nodes[lhsNodesSize].tensorObject->dimensions.begin(), result.nodes[lhsNodesSize].tensorObject->dimensions.end());
        result.nodes[lhsNodesSize].tensorObject->reinterpret_dimensions(dimensions);
        std::vector<TensorNode::Link> newLinks;
        newLinks.emplace_back(lhsNodesSize-1, linkPos, 1, false);
        newLinks.insert(newLinks.end(), result.nodes[lhsNodesSize].neighbors.begin(), result.nodes[lhsNodesSize].neighbors.end());
        result.nodes[lhsNodesSize].neighbors = newLinks;
        // links of nodes[lhsNodesSize] changed, so change the link of lhsNodesSize+1 correspondingly
        if (result.nodes.size()>lhsNodesSize+1) {
            result.nodes[lhsNodesSize+1].neighbors[0].indexPosition += 1;
        }

        // add all external indices of rhs
        result.externalLinks.clear();
        result.dimensions.clear();
        result.dimensions.push_back(_lhs.dimensions[0]);
        result.externalLinks.emplace_back(0, 0, _lhs.dimensions[0], false);
        for (size_t i=1; i<lhsNodesSize; ++i) {
            const size_t d=_lhs.dimensions[i];
            result.externalLinks.emplace_back(i, 1, d, false);
            result.dimensions.push_back(d);
        }
        for (size_t i=0; i<rhsNodesSize; ++i) {
            const size_t d=_rhs.dimensions[i];
            result.externalLinks.emplace_back(lhsNodesSize+i, 1, d, false);
            result.dimensions.push_back(d);
        }
        if (isOperator) {
            result.dimensions.push_back(_lhs.dimensions[lhsNodesSize]);
            result.externalLinks.emplace_back(0, 1, _lhs.dimensions[lhsNodesSize], false);
            for (size_t i=1; i<lhsNodesSize; ++i) {
                const size_t d=_lhs.dimensions[i];
                result.externalLinks.emplace_back(i, 2, d, false);
                result.dimensions.push_back(d);
            }
            for (size_t i=0; i<rhsNodesSize; ++i) {
                const size_t d=_rhs.dimensions[i];
                result.externalLinks.emplace_back(lhsNodesSize+i, 2, d, false);
                result.dimensions.push_back(d);
            }
        }
        
        REQUIRE(result.is_valid_tt(), "ie");
        return result;
    }
    
    template<bool isOperator>
    TTNetwork<isOperator> TTNetwork<isOperator>::dyadic_product(const std::vector<std::reference_wrapper<TTNetwork<isOperator>>> &_tensors) {
        if (_tensors.size() == 0) {
            return TTNetwork();
        } 
        TTNetwork result(_tensors.front());
        for (size_t i=0; i<_tensors.size(); ++i) {
            dyadic_product(result, _tensors[i]);
        }
        return result;
    }
    
    
    template<bool isOperator>
    std::pair<TensorNetwork, TensorNetwork> TTNetwork<isOperator>::chop(const size_t _position) const {
        REQUIRE(is_valid_tt(), "Invalid TT cannot be chopped.");
        CHECK(!isOperator, warning, "Chop is not yet testet for TTOperators"); //TODO test it!
        
        const size_t N = isOperator?2:1;
        const size_t numNodes = degree()/N;
        
        REQUIRE(_position < numNodes, "Can't spilt a " << numNodes << " node TTNetwork at position " << _position);
        
        // Create the resulting TNs
        TensorNetwork left, right;
        left.factor = 1;
        right.factor = 1;
        
        if(_position > 0) {
            for(size_t i = 0; i < _position; ++i) {
                left.dimensions.push_back(dimensions[i]);
                left.externalLinks.push_back(externalLinks[i]);
                left.nodes.push_back(nodes[i]);
            }
            if(isOperator) {
                for(size_t i = 0; i < _position; ++i) {
                    left.dimensions.push_back(dimensions[i+numNodes]);
                    left.externalLinks.push_back(externalLinks[i+numNodes]);
                }
            }
            left.dimensions.push_back(left.nodes.back().neighbors.back().dimension);
            left.externalLinks.emplace_back(_position-1, _position == 1 ? N : N+1, left.nodes.back().neighbors.back().dimension , false);
            left.nodes.back().neighbors.back().external = true;
            left.nodes.back().neighbors.back().indexPosition = isOperator ? 2*_position-1 :_position;
        }
        
        if(_position < numNodes-1) {
            right.dimensions.push_back(nodes[_position+1].neighbors.front().dimension);
            right.externalLinks.emplace_back(_position+1, 0, nodes[_position+1].neighbors.front().dimension , false); // NOTE other will be corrected to 0 in the following steps
            for(size_t i = _position+1; i < numNodes; ++i) {
                right.dimensions.push_back(dimensions[i]);
                right.externalLinks.push_back(externalLinks[i]);
                right.nodes.push_back(nodes[i]);
            }
            if(isOperator) {
                for(size_t i = _position+1; i < numNodes; ++i) {
                    right.dimensions.push_back(dimensions[i+numNodes]);
                    right.externalLinks.push_back(externalLinks[i+numNodes]);
                }
            }
            right.nodes.front().neighbors.front().external = true;
            right.nodes.front().neighbors.front().indexPosition = _position; // NOTE indexPosition will be corrected to 0 in the following steps
            
            // Account for the fact that the first _position+1 nodes do not exist
            for(TensorNode::Link& link : right.externalLinks) {
                link.other -= _position+1;
            }
            
            for(TensorNode& node : right.nodes) {
                for(TensorNode::Link& link : node.neighbors) {
                    if(link.external) {
                        link.indexPosition -= _position;
                    } else {
                        link.other -= _position+1;
                    }
                }
            }
        }
        
        REQUIRE(left.is_valid_network(), "Internal Error");
        REQUIRE(right.is_valid_network(), "Internal Error");
        
        return std::pair<TensorNetwork, TensorNetwork>(std::move(left), std::move(right));
    }
    
    
    template<bool isOperator>
    void TTNetwork<isOperator>::round(value_t _eps) {
        cannonicalize_left();
        const size_t N = isOperator?2:1;
        round_train(*this, std::vector<size_t>(degree()/N-1, size_t(-1)), _eps);
    }

    template<bool isOperator>
    void TTNetwork<isOperator>::round(size_t _maxRank) {
        cannonicalize_left();
        const size_t N = isOperator?2:1;
        round_train(*this, std::vector<size_t>(degree()/N-1 ,_maxRank), 1e-15);
    }

    template<bool isOperator>
    void TTNetwork<isOperator>::round(const std::vector<size_t> &_maxRanks) {
        cannonicalize_left();
        round_train(*this, _maxRanks, 1e-15);
    }
    
    template<bool isOperator>
    void TTNetwork<isOperator>::round(int _maxRank) {
        REQUIRE( _maxRank > 0, "MaxRank must be positive");
        round(size_t(_maxRank));
    }

    template<bool isOperator>
    std::vector<size_t> TTNetwork<isOperator>::ranks() const {
        std::vector<size_t> res;
        for (size_t n=0; n<nodes.size()-1; ++n) {
            res.push_back(nodes[n].neighbors.back().dimension);
        }
        return res;
    }
    
    template<bool isOperator>
    size_t TTNetwork<isOperator>::rank(size_t _i) const {
        REQUIRE(_i < nodes.size()-1, "requested illegal rank");
        return nodes[_i].neighbors.back().dimension;
    }
    
    template<bool isOperator>
    size_t TTNetwork<isOperator>::datasize() const {
        size_t result = 0;
        for (const TensorNode &n : nodes) {
            result += n.tensorObject->size;
        }
        return result;
    }
    
    template<bool isOperator>
    void TTNetwork<isOperator>::cannonicalize_left() {
        Index i,r,j;
        FullTensor core(2);
        for (size_t n=nodes.size()-1; n > 0; --n) {
            REQUIRE(!nodes[n].erased, "ie n="<<n);
            Tensor &currTensor = *nodes[n].tensorObject;
            ( core(j,r), currTensor(r,i&1) ) = RQ(currTensor(j,i&1));  //TODO we want a rank-detecting QR at this point?
            Tensor &nextTensor = *nodes[n-1].tensorObject;
            nextTensor(j&1,i) = nextTensor(j&1,r) * core(r,i);
            if (currTensor.dimensions[0] != nodes[n].neighbors.front().dimension) {
                nodes[n].neighbors.front().dimension = nodes[n-1].neighbors.back().dimension = currTensor.dimensions[0];
            }
        }
    }
    
    template<bool isOperator>
    void TTNetwork<isOperator>::cannonicalize_right() {
        Index i,r,j;
        FullTensor core(2);
        for (size_t n=0; n<nodes.size()-1; ++n) {
            Tensor &currTensor = *nodes[n].tensorObject;
            ( currTensor(i&1,r), core(r,j) ) = QR(currTensor(i&1,j)); //TODO we want a rank-detecting QR at this point?
            Tensor &nextTensor = *nodes[n+1].tensorObject;
            nextTensor(i,j&1) = core(i,r) * nextTensor(r,j&1);
            if (nextTensor.dimensions[0] != nodes[n+1].neighbors.front().dimension) {
                nodes[n+1].neighbors.front().dimension = nodes[n].neighbors.back().dimension = nextTensor.dimensions[0];
            }
        }
    }
    
    template<bool isOperator>
    TensorNetwork* TTNetwork<isOperator>::get_copy() const {
        return new TTNetwork(*this);
    }
    
    template<bool isOperator>
    value_t TTNetwork<isOperator>::frob_norm() const {
        REQUIRE(is_valid_tt(), "frob_norm of illegal TT");
        return nodes.back().tensorObject->frob_norm();
    }
    
    template<bool isOperator>
    bool TTNetwork<isOperator>::is_in_expected_format() const {
        return is_valid_tt();
    }
    
    
    /*- - - - - - - - - - - - - - - - - - - - - - - - - -  Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    template<bool isOperator>
    TTNetwork<isOperator>& TTNetwork<isOperator>::operator+=(const TTNetwork<isOperator>& _other) {
        Index i;
        (*this)(i&0) = (*this)(i&0) + _other(i&0);
        return *this;
    }
    
    template<bool isOperator>
    TTNetwork<isOperator>  TTNetwork<isOperator>::operator+(const TTNetwork<isOperator>& _other) const {
        TTNetwork cpy(*this);
        cpy += _other;
        return cpy;
    }
    
    template<bool isOperator>
    TTNetwork<isOperator>& TTNetwork<isOperator>::operator-=(const TTNetwork<isOperator>& _other) {
        Index i;
        (*this)(i&0) = (*this)(i&0) - _other(i&0);
        return *this;
    }
    
    template<bool isOperator>
    TTNetwork<isOperator>  TTNetwork<isOperator>::operator-(const TTNetwork<isOperator>& _other) const {
        TTNetwork cpy(*this);
        cpy -= _other;
        return cpy;
    }
    
    template<bool isOperator>
    TTNetwork<isOperator>& TTNetwork<isOperator>::operator*=(const value_t _prod) {
        factor *= _prod;
        return *this;
        
    }
    
    template<bool isOperator>
    TTNetwork<isOperator>  TTNetwork<isOperator>::operator*(const value_t _prod) const {
        TTNetwork result(*this);
        result *= _prod;
        return result;
    }
    
    template<bool isOperator>
    TTNetwork<isOperator>& TTNetwork<isOperator>::operator/=(const value_t _div) {
        factor /= _div;
        return *this;
    }
    
    template<bool isOperator>
    TTNetwork<isOperator>  TTNetwork<isOperator>::operator/(const value_t _div) const {
        TTNetwork result(*this);
        result /= _div;
        return result;
    }
    
    
    /*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    
    template<bool isOperator>
    bool TTNetwork<isOperator>::specialized_contraction(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) const {
        REQUIRE(!_out.tensorObject, "Internal Error.");
        
        // Only TTOperators construct stacks, so no specialized contractions for TTTensors
        if(!isOperator) { return false; }
        
        const std::vector<Index> myIndices = _me.get_assigned_indices();
        const std::vector<Index> otherIndices = _other.get_assigned_indices();
        
        bool otherTT = dynamic_cast<const TTTensor *>(_other.tensorObjectReadOnly) || dynamic_cast<const internal::TTStack<false> *>(_other.tensorObjectReadOnly);
        bool otherTO = !otherTT && (dynamic_cast<const TTOperator *>(_other.tensorObjectReadOnly) || dynamic_cast<const internal::TTStack<true> *>(_other.tensorObjectReadOnly));
        
        if (!otherTT && !otherTO) {
            return false;
        }
        
        // Determine my first half and second half of indices
        auto midIndexItr = myIndices.begin();
        size_t spanSum = 0;
        while (spanSum < _me.degree() / 2) {
            REQUIRE(midIndexItr != myIndices.end(), "ie");
            spanSum += midIndexItr->span;
            ++midIndexItr;
        }
        if (spanSum > _me.degree() / 2) {
            return false; // an index spanned some links of the left and some of the right side
        }
        
        if (otherTT) {
            // ensure fitting indices
            if (equal(myIndices.begin(), midIndexItr, otherIndices.begin(), otherIndices.end()) || equal(midIndexItr, myIndices.end(), otherIndices.begin(), otherIndices.end())) {
                TensorNetwork *res = new internal::TTStack<false>;
                *res = *_me.tensorObjectReadOnly;
                res->factor *= _other.tensorObjectReadOnly->factor;
                _out.reset(res, myIndices, true);
                TensorNetwork::add_network_to_network(_out, _other);
                return true;
            } else {
                return false;
            }
        } else { // other is operator or operator stack
            // determine other middle index
            auto otherMidIndexItr = otherIndices.begin();
            spanSum = 0;
            while (spanSum < _other.degree() / 2) {
                REQUIRE(otherMidIndexItr != otherIndices.end(), "ie");
                spanSum += otherMidIndexItr->span;
                ++otherMidIndexItr;
            }
            if (spanSum > _other.degree() / 2) {
                return false; // an index spanned some links of the left and some of the right side
            }
            // or indices in fitting order to contract the TTOs
            if (   equal(myIndices.begin(), midIndexItr, otherIndices.begin(), otherMidIndexItr) 
                || equal(midIndexItr, myIndices.end(), otherIndices.begin(), otherMidIndexItr)
                || equal(myIndices.begin(), midIndexItr, otherMidIndexItr, otherIndices.end()) 
                || equal(midIndexItr, myIndices.end(), otherMidIndexItr, otherIndices.end())    ) 
            {
                TensorNetwork *res = new internal::TTStack<true>;
                *res = *_me.tensorObjectReadOnly;
                res->factor *= _other.tensorObjectReadOnly->factor;
                _out.reset(res, myIndices, true);
                TensorNetwork::add_network_to_network(_out, _other);
                return true;
            } else {
                return false;
            }
        }
    }
    
    template<bool isOperator>
    bool TTNetwork<isOperator>::specialized_sum(IndexedTensorWritable<TensorNetwork> &_out, const IndexedTensorReadOnly<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) const {
        const size_t N = isOperator?2:1;
        
        const std::vector<Index> myIndices = _me.get_assigned_indices();
        const std::vector<Index> otherIndices = _other.get_assigned_indices();
        
        // If the indices are in different order, we are lost. TODO inverse order is also ok...
        if(myIndices != otherIndices) { return false; }
        REQUIRE(_me.tensorObjectReadOnly->dimensions == _other.tensorObjectReadOnly->dimensions, "TT sum requires both operants to share the same dimensions");
        
        // If the other is not a TT tensor (or stack) we are also lost
        const TTNetwork* _otherPtr = dynamic_cast<const TTNetwork*>( _other.tensorObjectReadOnly);
        if(!_otherPtr) { return false; }
        
        // TODO the order is not canonical, because if I am no Stack I don't have to know whether or not i am moveable
        // If I am in fact a TTTensorStack, we have to evaluate me to TTNetwork
        std::unique_ptr<IndexedTensor<TensorNetwork>> meStorage;
        const IndexedTensorReadOnly<TensorNetwork> *realMePtr = &_me;
        const IndexedTensorMoveable<TensorNetwork> *movMe = dynamic_cast<const IndexedTensorMoveable<TensorNetwork> *>(&_me);
        if (movMe) {
            internal::TTStack<isOperator> *stackMe = dynamic_cast<internal::TTStack<isOperator> *>(movMe->tensorObject);
            if (stackMe) {
                meStorage.reset(new IndexedTensor<TensorNetwork>(new TTNetwork(_me.degree()), myIndices, true));
                (*meStorage) = _me;
                realMePtr = meStorage.get();
            }
        } else {
            REQUIRE(!dynamic_cast<const internal::TTStack<isOperator> *>(_me.tensorObjectReadOnly),"ie - non-moveable TTStack detected");
        }
        const IndexedTensorReadOnly<TensorNetwork> &realMe = *realMePtr;
        
        // If other is in fact a TTTensorStack, we have to evaluate it to tttensor
        std::unique_ptr<IndexedTensor<TensorNetwork>> otherStorage;
        const IndexedTensorReadOnly<TensorNetwork> *realOtherPtr = &_other;
        const IndexedTensorMoveable<TensorNetwork> *movOther = dynamic_cast<const IndexedTensorMoveable<TensorNetwork> *>(&_other);
        if (movOther) {
            internal::TTStack<isOperator> *stackOther = dynamic_cast<internal::TTStack<isOperator> *>(movOther->tensorObject);
            if (stackOther) {
                otherStorage.reset(new IndexedTensor<TensorNetwork>(new TTNetwork(_other.degree()), otherIndices, true));
                (*otherStorage) = _other;
                realOtherPtr = otherStorage.get();
            }
        } else {
            REQUIRE(!dynamic_cast<const internal::TTStack<isOperator> *>(_other.tensorObjectReadOnly),"ie - non-moveable TTStack detected");
        }
        const IndexedTensorReadOnly<TensorNetwork> &realOther = *realOtherPtr;
        
        // Number of Nodes to create
        const size_t numNodes = realMe.degree()/N;
        
        TTNetwork* tmpPtr = new TTNetwork();
        tmpPtr->factor = 1.0;
        
        //The external dimensions are the same as the ones of the input
        tmpPtr->dimensions = realMe.tensorObjectReadOnly->dimensions;
        REQUIRE(realOther.tensorObjectReadOnly->dimensions == realMe.tensorObjectReadOnly->dimensions, "Internal Error");
        
        IndexedTensor<TensorNetwork> tmpOut(tmpPtr, myIndices, true);
        TTNetwork& outTensor = *static_cast<TTNetwork*>(tmpOut.tensorObject);
        
        
        // Create the externalLinks first, as we know their position in advance
        outTensor.externalLinks.emplace_back(0, 0, outTensor.dimensions[0], false);
        for(size_t i = 1; i < numNodes; ++i) {
            outTensor.externalLinks.emplace_back(i, 1, outTensor.dimensions[i], false);
        }
        if(N == 2) {
            outTensor.externalLinks.emplace_back(0, 1, outTensor.dimensions[numNodes], false);
            for(size_t i = 1; i < numNodes; ++i) {
                outTensor.externalLinks.emplace_back(i, 2, outTensor.dimensions[numNodes+i], false);
            }
        }
        
        
        if(realMe.degree() == N) {
            // Create the one Node
            std::unique_ptr<Tensor> nextTensor;
            if(realMe.tensorObjectReadOnly->nodes[0].tensorObject->is_sparse() && realOther.tensorObjectReadOnly->nodes[0].tensorObject->is_sparse()) { // Both Sparse
                nextTensor.reset(realMe.tensorObjectReadOnly->nodes[0].tensorObject->get_copy());
                nextTensor->factor *= realMe.tensorObjectReadOnly->factor;
                *static_cast<SparseTensor*>(nextTensor.get()) += realOther.tensorObjectReadOnly->factor*(*static_cast<SparseTensor*>(realOther.tensorObjectReadOnly->nodes[0].tensorObject.get()));
            } else { // Maximal one sparse
                if(realMe.tensorObjectReadOnly->nodes[0].tensorObject->is_sparse()){
                    nextTensor.reset(new FullTensor(*static_cast<SparseTensor*>(realMe.tensorObjectReadOnly->nodes[0].tensorObject.get())));
                } else {
                    nextTensor.reset(new FullTensor(*static_cast<FullTensor*>(realMe.tensorObjectReadOnly->nodes[0].tensorObject.get())));
                }
                nextTensor->factor *= realMe.tensorObjectReadOnly->factor;
                if(realOther.tensorObjectReadOnly->nodes[0].tensorObject->is_sparse()){
                    *static_cast<FullTensor*>(nextTensor.get()) += realOther.tensorObjectReadOnly->factor*static_cast<SparseTensor&>(*realOther.tensorObjectReadOnly->nodes[0].tensorObject.get());
                } else {
                    *static_cast<FullTensor*>(nextTensor.get()) += realOther.tensorObjectReadOnly->factor*static_cast<FullTensor&>(*realOther.tensorObjectReadOnly->nodes[0].tensorObject.get());
                }
            }
            
            outTensor.nodes.emplace_back(std::move(nextTensor));
            outTensor.nodes.back().neighbors.emplace_back(-1, 0, outTensor.dimensions[0], true);
            if(N == 2) { outTensor.nodes.back().neighbors.emplace_back(-1, 1, outTensor.dimensions[1], true); }
            _out.assign(std::move(tmpOut));
            return true;
        }
        
        for(size_t position = 0; position < numNodes; ++position) {
            // Get current input nodes
            // TODO sparse
            FullTensor &myNode = *static_cast<FullTensor*>(realMe.tensorObjectReadOnly->nodes[position].tensorObject.get());
            FullTensor &otherNode = *static_cast<FullTensor*>(realOther.tensorObjectReadOnly->nodes[position].tensorObject.get());
            
            // Structure has to be (for degree 4)
            // (L1 R1) * ( L2 0  ) * ( L3 0  ) * ( L4 )
            //           ( 0  R2 )   ( 0  R3 )   ( R4 )
            
            // Create a FullTensor for Node
            std::vector<size_t> nxtDimensions;
            if(position != 0) { 
                nxtDimensions.emplace_back(myNode.dimensions.front()+otherNode.dimensions.front());
            }
            nxtDimensions.emplace_back(outTensor.dimensions[position]);
            if(N == 2) { nxtDimensions.emplace_back(outTensor.dimensions[position+numNodes]); }
            if(position != numNodes-1) {
                nxtDimensions.emplace_back(myNode.dimensions.back()+otherNode.dimensions.back());
            }
            
            FullTensor* nxtTensor(new FullTensor(std::move(nxtDimensions)) ); // Ownership is given to the Node.
            
            
            // Create the Node
            outTensor.nodes.emplace_back(std::unique_ptr<Tensor>(nxtTensor));
            if(position != 0) { outTensor.nodes.back().neighbors.emplace_back(position-1, ((position == 1) ? 0:1)+N, nxtTensor->dimensions.front(), false); }
            outTensor.nodes.back().neighbors.emplace_back(-1, position, outTensor.dimensions[position], true);
            if(N == 2) { outTensor.nodes.back().neighbors.emplace_back(-1, position+numNodes, outTensor.dimensions[position+numNodes], true); }
            if(position != numNodes-1 ) { outTensor.nodes.back().neighbors.emplace_back(position+1, 0, nxtTensor->dimensions.back(), false); }

            const size_t leftIdxOffset = nxtTensor->size/nxtTensor->dimensions.front();
            const size_t extIdxOffset = nxtTensor->dimensions.back();
            const size_t myLeftIdxOffset = myNode.size/myNode.dimensions.front();
            const size_t myExtIdxOffset = myNode.dimensions.back();
            const size_t otherLeftIdxOffset = otherNode.size/otherNode.dimensions.front();
            const size_t otherExtIdxOffset = otherNode.dimensions.back();
            const size_t otherGeneralOffset = (position == 0 ? 0 : myNode.dimensions.front()*leftIdxOffset) + (position == numNodes-1 ? 0 : myNode.dimensions.back());
            
            
            
            // Copy own Tensor into place
            if(position == numNodes-1) {
                for(size_t leftIdx = 0; leftIdx < myNode.dimensions.front(); ++leftIdx) {
                    for(size_t extIdx = 0; extIdx < myNode.size/(myNode.dimensions.front()*myNode.dimensions.back()); ++extIdx) {
                        // RightIdx can be copied as one piece
                        array_scaled_copy(nxtTensor->data.get() + leftIdx*leftIdxOffset + extIdx*extIdxOffset, myNode.factor*realMe.tensorObjectReadOnly->factor, myNode.data.get() + leftIdx*myLeftIdxOffset + extIdx*myExtIdxOffset, myNode.dimensions.back());
                    }
                }
            } else {
                REQUIRE(!myNode.has_factor(), "Only Core node, which has to be the last node, is allowed to have a factor");
                for(size_t leftIdx = 0; leftIdx < myNode.dimensions.front(); ++leftIdx) {
                    for(size_t extIdx = 0; extIdx < myNode.size/(myNode.dimensions.front()*myNode.dimensions.back()); ++extIdx) {
                        // RightIdx can be copied as one piece
                        array_copy(nxtTensor->data.get() + leftIdx*leftIdxOffset + extIdx*extIdxOffset, myNode.data.get() + leftIdx*myLeftIdxOffset + extIdx*myExtIdxOffset, myNode.dimensions.back());
                    }
                }
            }
            
            
            // Copy other Tensor into place
            if(position == numNodes-1) {
                for(size_t leftIdx = 0; leftIdx < otherNode.dimensions.front(); ++leftIdx) {
                    for(size_t extIdx = 0; extIdx < otherNode.size/(otherNode.dimensions.front()*otherNode.dimensions.back()); ++extIdx) {
                        // RightIdx can be copied as one piece
                        array_scaled_copy(nxtTensor->data.get() + leftIdx*leftIdxOffset + extIdx*extIdxOffset + otherGeneralOffset, otherNode.factor*realOther.tensorObjectReadOnly->factor, otherNode.data.get() + leftIdx*otherLeftIdxOffset + extIdx*otherExtIdxOffset, otherNode.dimensions.back());
                    }
                }
            } else {
                REQUIRE(!otherNode.has_factor(), "Only Core node, which has to be the last node, is allowed to have a factor");
                for(size_t leftIdx = 0; leftIdx < otherNode.dimensions.front(); ++leftIdx) {
                    for(size_t extIdx = 0; extIdx < otherNode.size/(otherNode.dimensions.front()*otherNode.dimensions.back()); ++extIdx) {
                        // RightIdx can be copied as one piece
                        array_copy(nxtTensor->data.get() + leftIdx*leftIdxOffset + extIdx*extIdxOffset + otherGeneralOffset, otherNode.data.get() + leftIdx*otherLeftIdxOffset + extIdx*otherExtIdxOffset, otherNode.dimensions.back());
                    }
                }
            }
        }
        
        outTensor.cannonicalize_right();
        _out.assign(std::move(tmpOut));
        return true;
    }
    
    
    template<bool isOperator>
    void TTNetwork<isOperator>::specialized_evaluation(const IndexedTensorWritable<TensorNetwork> &_me, const IndexedTensorReadOnly<TensorNetwork> &_other) {
        const std::vector<Index> myIndices = _me.get_assigned_indices(_other.degree()); // TODO this wont work if we have fixed indices in TT tensors.
        const std::vector<Index> otherIndices = _other.get_assigned_indices();
        const size_t numNodes = _other.degree()/(isOperator ? 2 :1);
        
        REQUIRE(_me.tensorObject == this, "Internal Error.");
        
        // First check whether the other is a TTNetwork as well, otherwise we can skip to fallback
        const TTNetwork* otherTTN = dynamic_cast<const TTNetwork*>(_other.tensorObjectReadOnly);
        if(otherTTN) {
            // Check whether the index order coincides
            if(myIndices == otherIndices) {
                // Assign the other to me
                *_me.tensorObject = *otherTTN;
                
                // Check whether the other is a stack and needs to be contracted
                const internal::TTStack<isOperator> *otherTTS = dynamic_cast<const internal::TTStack<isOperator>*>(_other.tensorObjectReadOnly);
                if (otherTTS) {
                    contract_stack(_me);
                    static_cast<TTNetwork*>(_me.tensorObject)->cannonicalize_right(); // TODO cannonicalize_right should be called by contract_stack
                }
                return;
            }
            
            // For TTOperators check whether the index order is transpoed
            if(isOperator) {
                bool transposed = false;
                
                auto midIndexItr = myIndices.begin();
                size_t spanSum = 0;
                while (spanSum < numNodes) {
                    REQUIRE(midIndexItr != myIndices.end(), "Internal Error.");
                    spanSum += midIndexItr->span;
                    ++midIndexItr;
                }
                if (spanSum == numNodes) {
                    // tansposition possible on my end
                    auto otherMidIndexItr = otherIndices.begin();
                    spanSum = 0;
                    while (spanSum < numNodes) {
                        REQUIRE(otherMidIndexItr != otherIndices.end(), "Internal Error.");
                        spanSum += otherMidIndexItr->span;
                        ++otherMidIndexItr;
                    }
                    if (spanSum == numNodes) {
                        // other tensor also transposable
                        transposed = (equal(myIndices.begin(), midIndexItr, otherMidIndexItr, otherIndices.end())) 
                                    && (equal(midIndexItr, myIndices.end(), otherIndices.begin(), otherMidIndexItr));
                        
                    }
                }
                
                if(transposed) {
                    // Assign the other to me
                    *_me.tensorObject = *otherTTN;
                    
                    // Check whether the other is a stack and needs to be contracted
                    const internal::TTStack<isOperator> *otherTTS = dynamic_cast<const internal::TTStack<isOperator>*>(_other.tensorObjectReadOnly);
                    if (otherTTS) {
                        contract_stack(_me);
                        static_cast<TTNetwork*>(_me.tensorObject)->cannonicalize_right(); // TODO cannonicalize_right should be called by contract_stack
                    }
                    static_cast<TTNetwork<true>*>(_me.tensorObject)->transpose(); // NOTE: This cast is never called if isOperator is false.
                    return;
                }
            }
        }
         
        // Use FullTensor fallback
        if (_other.tensorObjectReadOnly->nodes.size() > 1) {
            LOG(warning, "assigning a general tensor network to TTOperator not yet implemented. casting to fullTensor first");
        }
        std::unique_ptr<Tensor> otherFull(_other.tensorObjectReadOnly->fully_contracted_tensor());
        std::unique_ptr<Tensor> otherReordered(otherFull->construct_new(otherFull->dimensions, DONT_SET_ZERO()));
        (*otherReordered)(myIndices) = (*otherFull)(otherIndices);
        
        // Cast to TTNetwork
        if(otherReordered->is_sparse()) {
            LOG(fatal, "Not yet implemented." ); //TODO
        } else {
            *_me.tensorObject = TTNetwork(std::move(*static_cast<FullTensor*>(otherReordered.get())));
        }
    }

    
    // Explicit instantiation of the two template parameters that will be implemented in the xerus library
    template class TTNetwork<false>;
    template class TTNetwork<true>;
}
