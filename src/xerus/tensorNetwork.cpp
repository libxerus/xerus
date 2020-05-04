// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2019 Benjamin Huber and Sebastian Wolf.
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
* @brief Implementation of the TensorNetwork class.
*/

#include <xerus/tensorNetwork.h>

#include <fstream>

#include <xerus/misc/stringUtilities.h>
#include <xerus/misc/containerSupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/missingFunctions.h>
#include <xerus/misc/fileIO.h>
#include <xerus/misc/internal.h>

#include <xerus/basic.h>
#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/indexedTensorList.h>
#include <xerus/indexedTensorMoveable.h>
#include <xerus/indexedTensor_tensor_factorisations.h>
#include <xerus/contractionHeuristic.h>

namespace xerus {
	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	TensorNetwork::TensorNetwork() {
		nodes.emplace_back(TensorNode(std::make_unique<Tensor>()));
	}


	TensorNetwork::TensorNetwork(Tensor _other) : dimensions(_other.dimensions) { //NOTE don't use std::move here, because we need _other to be untouched to move it later
		nodes.emplace_back(std::make_unique<Tensor>(std::move(_other)), init_from_dimension_array());
	}


	TensorNetwork::TensorNetwork( std::unique_ptr<Tensor>&& _tensor) : dimensions(_tensor->dimensions) {
		nodes.emplace_back(std::move(_tensor), init_from_dimension_array());
	}


	TensorNetwork::TensorNetwork(size_t _order) : dimensions(std::vector<size_t>(_order, 1)) {
		nodes.emplace_back(std::make_unique<Tensor>(std::vector<size_t>(_order, 1)), init_from_dimension_array());
	}


	TensorNetwork::TensorNetwork(const ZeroNode _nodeStatus) {
		if(_nodeStatus == ZeroNode::Add) {
			nodes.emplace_back(TensorNode(std::make_unique<Tensor>()));
		}
	}

	TensorNetwork* TensorNetwork::get_copy() const {
		return new TensorNetwork(*this);
	}


	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal Helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

	std::vector<TensorNetwork::Link> TensorNetwork::init_from_dimension_array() {
		std::vector<TensorNetwork::Link> newLinks;
		for (size_t d = 0; d < dimensions.size(); ++d) {
			externalLinks.emplace_back(0, d, dimensions[d], false);
			newLinks.emplace_back(-1, d, dimensions[d], true);
		}
		return newLinks;
	}


	TensorNetwork TensorNetwork::stripped_subnet(const std::function<bool(size_t)>& _idF) const {
		TensorNetwork cpy(ZeroNode::None);
		cpy.nodes.resize(nodes.size());
		cpy.dimensions = dimensions;
		cpy.externalLinks = externalLinks;
		for (size_t id = 0; id < nodes.size(); ++id) {
			if (!_idF(id)) { continue; }
			cpy.nodes[id] = nodes[id].strippped_copy();
			for (size_t i = 0; i < cpy.nodes[id].neighbors.size(); ++i) {
				TensorNetwork::Link &l = cpy.nodes[id].neighbors[i];
				if (!l.external) { // Link was not external before
					if (!_idF(l.other)) { // ...but is "external" to this subnet
						l.external = true;
						l.indexPosition = cpy.externalLinks.size();
						cpy.dimensions.emplace_back(l.dimension);
						cpy.externalLinks.emplace_back(id, i, l.dimension, false);
					}
				}
			}
		}

		size_t correction = 0;
		std::vector<long> toErase;
		for (size_t eid = 0; eid < cpy.externalLinks.size(); ++eid) {
			TensorNetwork::Link &l = cpy.externalLinks[eid];
			if (!_idF(l.other)) {
				toErase.emplace_back(long(eid));
				correction++;
			} else {
				INTERNAL_CHECK(cpy.nodes[l.other].neighbors[l.indexPosition].external, "ie");
				INTERNAL_CHECK(cpy.nodes[l.other].neighbors[l.indexPosition].indexPosition == eid, "ie");
				cpy.nodes[l.other].neighbors[l.indexPosition].indexPosition -= correction;
			}
		}

		for (size_t i = toErase.size(); i > 0; --i) {
			cpy.dimensions.erase(cpy.dimensions.begin()+toErase[i-1]);
			cpy.externalLinks.erase(cpy.externalLinks.begin()+toErase[i-1]);
		}

		cpy.require_valid_network(false);
		return cpy;
	}


	void TensorNetwork::contract_unconnected_subnetworks() {
		require_valid_network();

		if(order() == 0) {
			std::set<size_t> all;
			for(size_t i = 0; i < nodes.size(); ++i) { all.emplace_hint(all.end(), i); }
			contract(all);

		} else {
			std::vector<bool> seen(nodes.size(), false);
			std::vector<size_t> expansionStack;
			expansionStack.reserve(nodes.size());

			// Starting at every external link...
			for (const TensorNetwork::Link& el : externalLinks) {
				if(!seen[el.other]) {
					seen[el.other] = true;
					expansionStack.push_back(el.other);
				}
			}

			// ...traverse the connected nodes in a depth-first manner.
			while (!expansionStack.empty()) {
				const size_t curr = expansionStack.back();
				expansionStack.pop_back();

				// Add unseen neighbors
				for (const TensorNetwork::Link& n : nodes[curr].neighbors) {
					if ( !n.external && !seen[n.other] ) {
						seen[n.other] = true;
						expansionStack.push_back(n.other);
					}
				}
			}

			// Construct set of all unseen nodes...
			std::set<size_t> toContract;
			for (size_t i = 0; i < nodes.size(); ++i) {
				if (!seen[i]) {
					toContract.emplace_hint(toContract.end(), i);
				}
			}

			// ...and contract them
			if (!toContract.empty()) {
				const size_t remaining = contract(toContract);

				INTERNAL_CHECK(nodes[remaining].order() == 0, "Internal Error.");

				// Remove contracted order-0 tensor
				nodes[remaining].erased = true;
				for(size_t i = 0; i < nodes.size(); ++i) {
					if(!nodes[i].erased) {
						*nodes[i].tensorObject *= (*nodes[remaining].tensorObject)[0];
						break;
					}
					INTERNAL_CHECK(i < nodes.size()-1, "Internal Error.");
				}
			}
		}

		sanitize();

		INTERNAL_CHECK(!nodes.empty(), "Internal error");
	}


	std::pair< size_t, size_t > TensorNetwork::find_common_edge(const size_t _nodeA, const size_t _nodeB) const {
		size_t posA=~0ul, posB=~0ul;

		// Find common edge in nodeA
		IF_CHECK(bool foundCommon = false;)
		for(size_t i = 0; i < nodes[_nodeA].neighbors.size(); ++i) {
			if(nodes[_nodeA].neighbors[i].other == _nodeB) {
				posA = i;
				REQUIRE(!foundCommon, "TN round/move core does not work if the two nodes share more than one link.");
				IF_CHECK( foundCommon = true; )
				IF_NO_CHECK( break; )
			}
		}
		REQUIRE(foundCommon, "TN round does not work if the two nodes share no link.");

		posB = nodes[_nodeA].neighbors[posA].indexPosition;

		return std::pair<size_t, size_t>(posA, posB);
	}


	void TensorNetwork::perform_traces(const size_t _nodeId) {
		for (size_t i = 0; i < nodes[_nodeId].order(); ++i) {
			const TensorNetwork::Link &link = nodes[_nodeId].neighbors[i];
			if (link.links(_nodeId)) {
				nodes[_nodeId].tensorObject->perform_trace(i, link.indexPosition);

				const std::vector<Link> linkCopy(nodes[_nodeId].neighbors);

				for(size_t j = i+1; j < link.indexPosition; ++j) {
					const Link& otherLink = linkCopy[j];
					if(otherLink.external) {
						externalLinks[otherLink.indexPosition].indexPosition -= 1;
					} else {
						nodes[otherLink.other].neighbors[otherLink.indexPosition].indexPosition -= 1;
					}
				}

				for(size_t j = link.indexPosition+1; j < nodes[_nodeId].order(); ++j) {
					const Link& otherLink = linkCopy[j];
					if(otherLink.external) {
						externalLinks[otherLink.indexPosition].indexPosition -= 2;
					} else {
						nodes[otherLink.other].neighbors[otherLink.indexPosition].indexPosition -= 2;
					}
				}

				nodes[_nodeId].neighbors.erase(nodes[_nodeId].neighbors.begin() + link.indexPosition);
				nodes[_nodeId].neighbors.erase(nodes[_nodeId].neighbors.begin() + i);

				//Redo this index
				i -= 1;
			}
		}
	}


	void TensorNetwork::sanitize() {
		std::vector<size_t> idMap(nodes.size());
		
		// Move nodes
		size_t newId = 0, oldId = 0;
		for (; oldId < nodes.size(); ++oldId) {
			if (!nodes[oldId].erased) {
				idMap[oldId] = newId;
				if (newId != oldId) { std::swap(nodes[newId], nodes[oldId]); }
				newId++;
			}
		}
		
		// Update links
		nodes.resize(newId);
		for (TensorNode &n : nodes) {
			for (TensorNetwork::Link &l : n.neighbors) {
				if (!l.external) { l.other = idMap[l.other]; }
			}
		}
		
		// Update external links
		for (TensorNetwork::Link &l : externalLinks) {
			l.other = idMap[l.other];
		}
	}



	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Conversions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	TensorNetwork::operator Tensor() const {
		require_valid_network();

		std::set<size_t> all;
		for(size_t i = 0; i < nodes.size(); ++i) { all.emplace_hint(all.end(), i); }

		TensorNetwork cpy(*this);
		size_t res = cpy.contract(all);

		std::vector<size_t> shuffle(order());
		for(size_t i = 0; i < cpy.nodes[res].neighbors.size(); ++i) {
			INTERNAL_CHECK(cpy.nodes[res].neighbors[i].external, "Internal Error");
			shuffle[i] = cpy.nodes[res].neighbors[i].indexPosition;
		}
		Tensor result;

		reshuffle(result, *cpy.nodes[res].tensorObject, shuffle);

		return result;
	}


	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Access - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	value_t TensorNetwork::operator[](const size_t _position) const {
		require_valid_network();

		if (order() == 0) {
			REQUIRE(_position == 0, "Tried to access non-existing entry of TN");
			value_t value = 1.0;
			for(const TensorNode& node : nodes) { value *= (*node.tensorObject)[0]; }
			return value;
		}

		std::vector<size_t> positions(order());
		size_t remains = _position;
		for(size_t i = order(); i > 1; --i) {
			positions[i-1] = remains%dimensions[i-1];
			remains /= dimensions[i-1];
		}
		positions[0] = remains;
		return operator[](positions);
	}


	value_t TensorNetwork::operator[](const Tensor::MultiIndex& _positions) const {
		require_valid_network();

		TensorNetwork partialCopy;
		partialCopy.nodes = nodes;

		// Set all external indices in copy to the fixed values and evaluate the tensorObject accordingly
		for(TensorNode& node : partialCopy.nodes) {
			// Fix slates in external links
			size_t killedDimensions = 0;
			for(size_t i = 0; i < node.neighbors.size(); ++i) {
				if(node.neighbors[i].external) {
					node.tensorObject->fix_mode(i-killedDimensions, _positions[node.neighbors[i].indexPosition]);
					killedDimensions++;
				}
			}

			// Remove all external links, because they don't exist anymore
			node.neighbors.erase(std::remove_if(node.neighbors.begin(), node.neighbors.end(), [](const TensorNetwork::Link& _test){return _test.external;}), node.neighbors.end());

			// Adjust the Links
			for(size_t i = 0; i < node.neighbors.size(); ++i) {
				partialCopy.nodes[node.neighbors[i].other].neighbors[node.neighbors[i].indexPosition].indexPosition = i;
			}
		}

		// Contract the complete network (there are not external Links)
		partialCopy.contract_unconnected_subnetworks();

		INTERNAL_CHECK(partialCopy.nodes.size() == 1, "Internal Error.");

		return (*partialCopy.nodes[0].tensorObject)[0];
	}


	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

	void TensorNetwork::operator*=(const value_t _factor) {
		REQUIRE(!nodes.empty(), "There must not be a TTNetwork without any node");
		REQUIRE(!nodes[0].erased, "There must not be an erased node.");
		*nodes[0].tensorObject *= _factor;
	}


	void TensorNetwork::operator/=(const value_t _divisor) {
		REQUIRE(!nodes.empty(), "There must not be a TTNetwork without any node");
		REQUIRE(!nodes[0].erased, "There must not be an erased node.");
		*nodes[0].tensorObject /= _divisor;
	}


	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Indexing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
	internal::IndexedTensor<TensorNetwork> TensorNetwork::operator()(const std::vector<Index> & _indices) {
		return internal::IndexedTensor<TensorNetwork>(this, _indices, false);
	}


	internal::IndexedTensor<TensorNetwork> TensorNetwork::operator()(     std::vector<Index>&& _indices) {
		return internal::IndexedTensor<TensorNetwork>(this, std::move(_indices), false);
	}


	internal::IndexedTensorReadOnly<TensorNetwork> TensorNetwork::operator()(const std::vector<Index> & _indices) const {
		return internal::IndexedTensorReadOnly<TensorNetwork>(this, _indices);
	}


	internal::IndexedTensorReadOnly<TensorNetwork> TensorNetwork::operator()(     std::vector<Index>&& _indices) const {
		return internal::IndexedTensorReadOnly<TensorNetwork>(this, std::move(_indices));
	}


	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */
	bool TensorNetwork::specialized_contraction(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>&  /*_out*/, internal::IndexedTensorReadOnly<TensorNetwork>&&  /*_me*/ , internal::IndexedTensorReadOnly<TensorNetwork>&&  /*_other*/ ) const {
		return false; // A general tensor Network can't do anything specialized
	}


	bool TensorNetwork::specialized_sum(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>&  /*_out*/, internal::IndexedTensorReadOnly<TensorNetwork>&&  /*_me*/, internal::IndexedTensorReadOnly<TensorNetwork>&&  /*_other*/) const {
		return false; // A general tensor Network can't do anything specialized
	}


	void TensorNetwork::specialized_evaluation(internal::IndexedTensorWritable<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) {
		TensorNetwork& me = *_me.tensorObject;
		const TensorNetwork& other = *_other.tensorObjectReadOnly;

		me = other;

		_other.assign_indices();

		link_traces_and_fix((me)(_other.indices));

		_me.assign_indices();

		// Needed if &me == &other
		const std::vector<size_t> otherDimensions = other.dimensions;
		const std::vector<Link> otherExtLinks = other.externalLinks;

		for (size_t i = 0, dimPosA = 0; i < _me.indices.size(); dimPosA += _me.indices[i].span, ++i) {
			size_t j = 0, dimPosB = 0;
			while (_me.indices[i] != _other.indices[j]) {
				dimPosB += _other.indices[j].span;
				++j;
				REQUIRE( j < _other.indices.size(), "LHS Index " << _me.indices[i] << " not found in RHS " << _other.indices);
			}

			REQUIRE(_me.indices[i].span == _other.indices[j].span, "Index spans must coincide");

			for (size_t s = 0; s < _me.indices[i].span; ++s) {
				me.dimensions[dimPosA+s] = otherDimensions[dimPosB+s];
				me.externalLinks[dimPosA+s] = otherExtLinks[dimPosB+s];
				me.nodes[me.externalLinks[dimPosA+s].other].neighbors[me.externalLinks[dimPosA+s].indexPosition].indexPosition = dimPosA+s;
				me.nodes[me.externalLinks[dimPosA+s].other].neighbors[me.externalLinks[dimPosA+s].indexPosition].dimension = me.dimensions[dimPosA+s];
			}
		}
	}

	/*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

	size_t TensorNetwork::degree() const {
		return order();
	}
	
	
	size_t TensorNetwork::order() const {
		return dimensions.size();
	}
	

	size_t TensorNetwork::datasize() const {
		size_t result = 0;
		for (const TensorNode& node : nodes) {
			result += node.tensorObject->size;
		}
		return result;
	}

	void TensorNetwork::reshuffle_nodes(const std::function<size_t(size_t)>& _f) {
		std::vector<TensorNode> newNodes(nodes.size());
		size_t newSize = 0;
		for (size_t i = 0; i < nodes.size(); ++i) {
			if (nodes[i].erased) { continue; }
			const size_t newIndex = _f(i);
			newSize = std::max(newSize, newIndex+1);
			REQUIRE(newNodes[newIndex].erased, "Tried to shuffle two nodes to the same new position " << newIndex << " i= " << i);
			newNodes[newIndex] = nodes[i];
			for (TensorNetwork::Link &l : newNodes[newIndex].neighbors) {
				if (!l.external) { l.other = _f(l.other); }
			}
		}
		nodes = newNodes;
		nodes.resize(newSize);

		for (auto &link : externalLinks) {
			link.other = _f(link.other);
		}
	}

#ifndef XERUS_DISABLE_RUNTIME_CHECKS
	void TensorNetwork::require_valid_network(const bool _check_erased) const {
		REQUIRE(externalLinks.size() == dimensions.size(), "externalLinks.size() != dimensions.size()");
		REQUIRE(!nodes.empty(), "There must always be at least one node!");

		// Per external link
		for (size_t n = 0; n < externalLinks.size(); ++n) {
			const TensorNetwork::Link& el = externalLinks[n];
			REQUIRE(el.other < nodes.size(), "External link " << n << " is inconsitent. The linked node " << el.other << " does not exist, as there are only " << nodes.size() << " nodes.");
			REQUIRE(el.dimension > 0, "External link " << n << " is corrupted. The link specifies zero as dimension.");
			REQUIRE(el.dimension == dimensions[n], "External link " << n << " is inconsitent. The specified dimension " << el.dimension << " does not match the " << n << "-th dimension of the Network, which is " << dimensions[n] << ".");
			REQUIRE(!el.external, "External link " << n << " is corrupted. It specifies itself to be external, but must link to a node.");

			const TensorNode& otherNode = nodes[el.other];
			const TensorNetwork::Link& otherLink = otherNode.neighbors[el.indexPosition];
			REQUIRE(otherNode.order() > el.indexPosition, "External link " << n << " is inconsitent. The link points to node " << el.other << " at IP " << el.indexPosition << ", but the target node only has " << otherNode.order() << " links.");
			REQUIRE(otherLink.external, "External link " << n << " is inconsitent. The link points to node " << el.other << " at IP " << el.indexPosition << ", but the target link says it is not external.");
			REQUIRE(otherLink.indexPosition == n, "External link " << n << " is inconsitent. The link points to node " << el.other << " at IP " << el.indexPosition << ", but the nodes link points to IP " << otherLink.indexPosition << " instead of " << n << ".");
			REQUIRE(otherLink.dimension == el.dimension, "External link " << n << " is inconsitent. The link points to node " << el.other << " at IP " << el.indexPosition << ". The dimension specified by the external link is " << el.dimension << " but the one of the target link is " << otherLink.dimension << ".");
		}

		// Per node
		for (size_t n = 0; n < nodes.size(); ++n) {
			const TensorNode &currNode = nodes[n];
			REQUIRE(!_check_erased || !currNode.erased, "Node " << n << " is marked erased, although this was not allowed.");
			if (currNode.tensorObject) {
				REQUIRE(currNode.order() == currNode.tensorObject->order(), "Node " << n << " has is inconsitent, as its tensorObject has order " << currNode.tensorObject->order() << " but there are " << currNode.order() << " links.");
			}

			// Per neighbor
			for (size_t i = 0; i < currNode.neighbors.size(); ++i) {
				const TensorNetwork::Link &el = currNode.neighbors[i];
				REQUIRE(el.dimension > 0, "n=" << n << " i=" << i);
				if (currNode.tensorObject) {
					REQUIRE(el.dimension == currNode.tensorObject->dimensions[i],  "n=" << n << " i=" << i << " " << el.dimension << " vs " << currNode.tensorObject->dimensions[i]);
				}

				if(!el.external) { // externals were already checked
					REQUIRE(el.other < nodes.size(), "Inconsitent Link from node " << n << " to node " << el.other << " from IP " << i << " to IP " << el.indexPosition << ". The target node does not exist, as there are only " << nodes.size() << " nodes.");
					const TensorNode &other = nodes[el.other];
					REQUIRE(other.order() > el.indexPosition, "Inconsitent Link from node " << n << " to node " << el.other << " from IP " << i << " to IP " << el.indexPosition << ". Link at target does not exist as there are only " << other.order() << " links.");
					REQUIRE(!other.neighbors[el.indexPosition].external, "Inconsitent Link from node " << n << " to node " << el.other << " from IP " << i << " to IP " << el.indexPosition << ". Link at target says it is external.");
					REQUIRE(other.neighbors[el.indexPosition].other == n, "Inconsitent Link from node " << n << " to node " << el.other << " from IP " << i << " to IP " << el.indexPosition << ". Link at target links to node " << other.neighbors[el.indexPosition].other << " at IP " << other.neighbors[el.indexPosition].indexPosition);
					REQUIRE(other.neighbors[el.indexPosition].indexPosition == i, "Inconsitent Link from node " << n << " to node " << el.other << " from IP " << i << " to IP " << el.indexPosition << ". Link at target links to node " << other.neighbors[el.indexPosition].other << " at IP " << other.neighbors[el.indexPosition].indexPosition);
					REQUIRE(other.neighbors[el.indexPosition].dimension == el.dimension, "Inconsitent Link from node " << n << " to node " << el.other << " from IP " << i << " to IP " << el.indexPosition << ". Dimension of this link is " << el.dimension << " but the target link says the dimension is " << other.neighbors[el.indexPosition].dimension);
				}
			}
		}
	}
#else
	/// No checks are performed with disabled checks...
	void TensorNetwork::require_valid_network(bool _check_erased) const { }
#endif

	void TensorNetwork::require_correct_format() const {
		require_valid_network();
	}


	void TensorNetwork::swap_external_links(const size_t _i, const size_t _j) {
		const TensorNetwork::Link& li = externalLinks[_i];
		const TensorNetwork::Link& lj = externalLinks[_j];
		nodes[li.other].neighbors[li.indexPosition].indexPosition = _j;
		nodes[lj.other].neighbors[lj.indexPosition].indexPosition = _i;
		std::swap(externalLinks[_i], externalLinks[_j]);
		std::swap(dimensions[_i], dimensions[_j]);
	}


	void TensorNetwork::add_network_to_network(internal::IndexedTensorWritable<TensorNetwork>&& _base, internal::IndexedTensorReadOnly<TensorNetwork>&& _toInsert) {
		_base.assign_indices();
		_toInsert.assign_indices();

		TensorNetwork &base = *_base.tensorObject;
		const TensorNetwork &toInsert = *_toInsert.tensorObjectReadOnly;

		const size_t firstNew = base.nodes.size();
		const size_t firstNewExternal = base.externalLinks.size();

		// Insert everything
		_base.indices.insert(_base.indices.end(), _toInsert.indices.begin(), _toInsert.indices.end());
		base.dimensions.insert(base.dimensions.end(), toInsert.dimensions.begin(), toInsert.dimensions.end());
		base.externalLinks.insert(base.externalLinks.end(), toInsert.externalLinks.begin(), toInsert.externalLinks.end());
		base.nodes.insert(base.nodes.end(), toInsert.nodes.begin(), toInsert.nodes.end());

		IF_CHECK (
			for (const Index &idx : _base.indices) {
				REQUIRE(misc::count(_base.indices, idx) < 3, "Index must not appear three (or more) times.");
			}
		)

		// Sanitize the externalLinks
		for (size_t i = firstNewExternal; i < base.externalLinks.size(); ++i) {
			base.externalLinks[i].other += firstNew;
		}

		// Sanitize the nodes (treating all external links as new external links)
		for (size_t i = firstNew; i < base.nodes.size(); ++i) {
			for(TensorNetwork::Link &l : base.nodes[i].neighbors) {
				if (!l.external) { // Link inside the added network
					l.other += firstNew;
				} else { // External link
					l.indexPosition += firstNewExternal;
				}
			}
		}

		// Find traces (former contractions may have become traces due to the joining)
		link_traces_and_fix(std::move(_base));

		_base.tensorObject->require_valid_network();
	}


	void TensorNetwork::link_traces_and_fix(internal::IndexedTensorWritable<TensorNetwork>&& _base) {
		TensorNetwork &base = *_base.tensorObject;
		_base.assign_indices();

		base.require_valid_network();

		IF_CHECK( std::set<Index> contractedIndices; )

		size_t passedDegree = 0;
		for(size_t i = 0; i < _base.indices.size(); ) {
			const Index& idx = _base.indices[i];

			// Search for a second occurance
			size_t j = i+1;
			size_t passedDegreeSecond = passedDegree + idx.span;
			for( ; j < _base.indices.size(); passedDegreeSecond += _base.indices[j].span, ++j) {
				if(idx == _base.indices[j]) { break; }
			}

			if(j < _base.indices.size()) { // There is a second occurance.
				REQUIRE(!misc::contains(contractedIndices, idx), "Indices must occur at most twice per contraction");
				REQUIRE(idx.span == _base.indices[j].span, "Index spans do not coincide " << idx << " vs " << _base.indices[j]);
				IF_CHECK( contractedIndices.insert(idx); )

				for (size_t n = 0; n < idx.span; ++n) {
					const TensorNetwork::Link &link1 = base.externalLinks[passedDegree+n];
					const TensorNetwork::Link &link2 = base.externalLinks[passedDegreeSecond+n];
					REQUIRE(link1.dimension == link2.dimension, "Index dimensions do not coincide: ["<<n<<"] " << link1.dimension << " vs " << link2.dimension << " Indices are " << idx << " and " << _base.indices[j] << " from " << _base.indices);

					base.nodes[link1.other].neighbors[link1.indexPosition] = link2;
					base.nodes[link2.other].neighbors[link2.indexPosition] = link1;
				}

				// Remove external links and dimensions from network
				base.externalLinks.erase(base.externalLinks.begin()+long(passedDegreeSecond), base.externalLinks.begin()+long(passedDegreeSecond+idx.span)); // note that passedDegreeSecond > passedDegree
				base.externalLinks.erase(base.externalLinks.begin()+long(passedDegree), base.externalLinks.begin()+long(passedDegree+idx.span));
				base.dimensions.erase(base.dimensions.begin()+long(passedDegreeSecond), base.dimensions.begin()+long(passedDegreeSecond+idx.span));
				base.dimensions.erase(base.dimensions.begin()+long(passedDegree), base.dimensions.begin()+long(passedDegree+idx.span));

				// Sanitize external links
				for(size_t k = passedDegree; k < passedDegreeSecond-idx.span; ++k) {
					base.nodes[base.externalLinks[k].other].neighbors[base.externalLinks[k].indexPosition].indexPosition -= idx.span;
				}

				for(size_t k = passedDegreeSecond-idx.span; k < base.externalLinks.size(); ++k) {
					base.nodes[base.externalLinks[k].other].neighbors[base.externalLinks[k].indexPosition].indexPosition -= 2*idx.span;
				}

				// Remove indices
				_base.indices.erase(_base.indices.begin()+j);
				_base.indices.erase(_base.indices.begin()+i);
			} else {
				passedDegree += idx.span;
				++i;
			}
		}

		// Apply fixed indices
		passedDegree = 0;
		for(size_t i = 0; i < _base.indices.size(); ) {
			const Index& idx = _base.indices[i];

			if(idx.fixed()) {
				// Fix the slates
				for(size_t k = passedDegree; k < passedDegree+idx.span; ++k) {
					base.fix_mode(passedDegree, idx.fixed_position());
				}

				// Remove index
				_base.indices.erase(_base.indices.begin()+i);
			} else {
				passedDegree += idx.span;
				++i;
			}
		}

		base.contract_unconnected_subnetworks();
	}


	void TensorNetwork::round_edge(const size_t _nodeA, const size_t _nodeB, const size_t _maxRank, const double _eps, const double _softThreshold) {
		require_valid_network();

		size_t fromPos, toPos;
		std::tie(fromPos, toPos) = find_common_edge(_nodeA, _nodeB);

		Tensor& fromTensor = *nodes[_nodeA].tensorObject;
		Tensor& toTensor = *nodes[_nodeB].tensorObject;

		const size_t fromDegree = fromTensor.order();
		const size_t toDegree = toTensor.order();

		const size_t currRank = fromTensor.dimensions[fromPos];

		// Reshuffle From if nessecary
		const bool transFrom = (fromPos == 0);
		const bool reshuffleFrom = (!transFrom && fromPos != fromDegree-1);
		std::vector<size_t> forwardShuffleFrom;
		std::vector<size_t> backwardShuffleFrom;
		if(reshuffleFrom) {
			forwardShuffleFrom.resize(fromDegree);
			backwardShuffleFrom.resize(fromDegree);

			for(size_t i = 0; i < fromPos; ++i) {
				forwardShuffleFrom[i] = i;
				backwardShuffleFrom[i] = i;
			}

			for(size_t i = fromPos; i+1 < fromDegree; ++i) {
				forwardShuffleFrom[i+1] = i;
				backwardShuffleFrom[i] = i+1;
			}

			forwardShuffleFrom[fromPos] = fromDegree-1;
			backwardShuffleFrom[fromDegree-1] = fromPos;

			reshuffle(fromTensor, fromTensor, forwardShuffleFrom);
		}

		// Reshuffle To if nessecary
		const bool transTo = (toPos == toDegree-1);
		const bool reshuffleTo = (!transTo && toPos != 0);
		std::vector<size_t> forwardShuffleTo;
		std::vector<size_t> backwardShuffleTo;
		if(reshuffleTo) {
			forwardShuffleTo.resize(toDegree);
			backwardShuffleTo.resize(toDegree);

			for(size_t i = 0; i < toPos; ++i) {
				forwardShuffleTo[i] = i+1;
				backwardShuffleTo[i+1] = i;
			}

			for(size_t i = toPos+1; i < toDegree; ++i) {
				forwardShuffleTo[i] = i;
				backwardShuffleTo[i] = i;
			}

			forwardShuffleTo[toPos] = 0;
			backwardShuffleTo[0] = toPos;

			reshuffle(toTensor, toTensor, forwardShuffleTo);
		}

		Tensor X, S;

		// Check whether prior QR makes sense
		if (5*fromTensor.size*toTensor.size >= 6*misc::pow(currRank, 4) ) {
			// Seperate the cores ...
			Tensor coreA, coreB;
			if(transFrom) {
				calculate_cq(coreA, fromTensor, fromTensor, 1);
			} else {
				calculate_qc(fromTensor, coreA, fromTensor, fromDegree-1);
			}

			if(transTo) {
				calculate_qc(toTensor, coreB, toTensor, toDegree-1);
			} else {
				calculate_cq(coreB, toTensor, toTensor, 1);
			}

			// ... contract them ...
			xerus::contract(X, coreA, transFrom, coreB, transTo, 1);

			// ... calculate svd ...
			calculate_svd(coreA, S, coreB, X, 1, _maxRank, _eps);

			if(_softThreshold > 0.0) {
				// Reduce SVs and find new rank
				size_t newRank = 0;
				do {
					S[{newRank, newRank}] = std::max(0.0, S[{newRank, newRank}]- _softThreshold);
					newRank++;
				} while (newRank < S.dimensions[0] && S[{newRank, newRank}] > _softThreshold);
				
				// Resize Tensors accordingly
				S.resize_mode(0, newRank);
				S.resize_mode(1, newRank);
				
				coreA.resize_mode(1, newRank);
				coreB.resize_mode(0, newRank);
				XERUS_REQUIRE_TEST;
			}

			// ... contract S to the right ...
			xerus::contract(coreB, S, false, coreB, false, 1);

			// ... and contract the cores back to their origins.
			if(transFrom) {
				xerus::contract(fromTensor, coreA, true, fromTensor, false, 1);
			} else {
				xerus::contract(fromTensor, fromTensor, false, coreA, false, 1);
			}

			if(transTo) {
				xerus::contract(toTensor, toTensor, false, coreB, true, 1);
			} else {
				xerus::contract(toTensor, coreB, false, toTensor, false, 1);
			}
		} else {
			xerus::contract(X, fromTensor, transFrom, toTensor, transTo, 1);

			calculate_svd(fromTensor, S, toTensor, X, fromDegree-1, _maxRank, _eps);
			
			if(_softThreshold > 0.0) {
				// Reduce SVs and find new rank
				size_t newRank = 0;
				do {
					S[{newRank, newRank}] = std::max(0.0, S[{newRank, newRank}]- _softThreshold);
					newRank++;
				} while (newRank < S.dimensions[0] && S[{newRank, newRank}] > _softThreshold);
				
				// Resize Tensors accordingly
				S.resize_mode(0, newRank);
				S.resize_mode(1, newRank);
				
				fromTensor.resize_mode(fromDegree-1, newRank);
				toTensor.resize_mode(0, newRank);
				XERUS_REQUIRE_TEST;
			}

			if(transTo) {
				xerus::contract(toTensor, toTensor, true, S, true, 1);
			} else {
				xerus::contract(toTensor, S, false, toTensor, false, 1);
			}

			if(transFrom) {
				backwardShuffleFrom.resize(fromDegree);
				for(size_t i = 0; i+1 < fromDegree; ++i) {
					backwardShuffleFrom[i] = i+1;
				}
				backwardShuffleFrom[fromDegree-1] = 0;
				reshuffle(fromTensor, fromTensor, backwardShuffleFrom);
			}
		}


		if(reshuffleFrom) {
			reshuffle(fromTensor, fromTensor, backwardShuffleFrom);
		}

		if(reshuffleTo) {
			reshuffle(toTensor, toTensor, backwardShuffleTo);
		}

		// Set the new dimension in the nodes
		nodes[_nodeA].neighbors[fromPos].dimension = nodes[_nodeA].tensorObject->dimensions[fromPos];
		nodes[_nodeB].neighbors[toPos].dimension = nodes[_nodeB].tensorObject->dimensions[toPos];
	}


	void TensorNetwork::transfer_core(const size_t _from, const size_t _to, const bool _allowRankReduction) {
		REQUIRE(_from < nodes.size() && _to < nodes.size(), " Illegal node IDs " << _from << "/" << _to << " as there are only " << nodes.size() << " nodes");
		require_valid_network();

		Tensor Q, R;
		size_t posA, posB;
		std::tie(posA, posB) = find_common_edge(_from, _to);

		Tensor& fromTensor = *nodes[_from].tensorObject;
		Tensor& toTensor = *nodes[_to].tensorObject;

		bool transR = false;
		if(posA == 0) {
			if(_allowRankReduction) {
				calculate_cq(R, Q, fromTensor, 1);
			} else {
				calculate_rq(R, Q, fromTensor, 1);
			}
			fromTensor = Q;
			transR = true;
		} else if(posA == nodes[_from].order()-1) {
			if(_allowRankReduction) {
				calculate_qc(Q, R, fromTensor, nodes[_from].order()-1);
			} else {
				calculate_qr(Q, R, fromTensor, nodes[_from].order()-1);
			}
			fromTensor = Q;
		} else {
			std::vector<size_t> forwardShuffle(nodes[_from].order());
			std::vector<size_t> backwardShuffle(nodes[_from].order());

			for(size_t i = 0; i < posA; ++i) {
				forwardShuffle[i] = i;
				backwardShuffle[i] = i;
			}

			for(size_t i = posA; i+1 < nodes[_from].order(); ++i) {
				forwardShuffle[i+1] = i;
				backwardShuffle[i] = i+1;
			}

			forwardShuffle[posA] = nodes[_from].order()-1;
			backwardShuffle[nodes[_from].order()-1] = posA;

			reshuffle(fromTensor, fromTensor, forwardShuffle);

			if(_allowRankReduction) {
				calculate_qc(Q, R, fromTensor, nodes[_from].order()-1);
			} else {
				calculate_qr(Q, R, fromTensor, nodes[_from].order()-1);
			}

			reshuffle(fromTensor, Q, backwardShuffle);
		}

		if( posB == 0 ) {
			xerus::contract(toTensor, R, transR, toTensor, false, 1);

		} else if( posB == nodes[_to].order()-1 ) {
			xerus::contract(toTensor, toTensor, false, R, !transR, 1);

		} else {
			std::vector<size_t> forwardShuffle(nodes[_to].order());
			std::vector<size_t> backwardShuffle(nodes[_to].order());

			for(size_t i = 0; i < posB; ++i) {
				forwardShuffle[i] = i+1;
				backwardShuffle[i+1] = i;
			}

			for(size_t i = posB+1; i < nodes[_to].order(); ++i) {
				forwardShuffle[i] = i;
				backwardShuffle[i] = i;
			}

			forwardShuffle[posB] = 0;
			backwardShuffle[0] = posB;

			reshuffle(toTensor, toTensor, forwardShuffle);

			xerus::contract(toTensor, R, posA == 0, toTensor, false, 1);

			reshuffle(toTensor, toTensor, backwardShuffle);
		}

		// Set the new dimension in the nodes
		nodes[_from].neighbors[posA].dimension = fromTensor.dimensions[posA];
		nodes[_to].neighbors[posB].dimension = toTensor.dimensions[posB];
	}


	void TensorNetwork::fix_mode(const size_t _mode, const size_t _slatePosition) {
		require_valid_network();

		REQUIRE(_mode < order(), "Invalid dimension to remove");
		REQUIRE(_slatePosition < dimensions[_mode], "Invalide _slatePosition to choose");

		const size_t extNode = externalLinks[_mode].other;
		const size_t extNodeIndexPos = externalLinks[_mode].indexPosition;

		// Correct the nodes external links
		for(size_t i = _mode+1; i < dimensions.size(); ++i) {
			REQUIRE(nodes[externalLinks[i].other].neighbors[externalLinks[i].indexPosition].indexPosition > 0, "Woo");
			nodes[externalLinks[i].other].neighbors[externalLinks[i].indexPosition].indexPosition--;
		}

		externalLinks.erase(externalLinks.begin()+_mode);
		dimensions.erase(dimensions.begin()+_mode);

		// Correct the others links of the affected node.
		for(size_t i = extNodeIndexPos+1; i < nodes[extNode].neighbors.size(); ++i) {
			const Link& link = nodes[extNode].neighbors[i];
			if(link.external) {
				externalLinks[link.indexPosition].indexPosition--;
			} else {
				// Check critical self links (i.e. where index position was allready modified).
				if(link.other == extNode && link.indexPosition+1 > extNodeIndexPos && link.indexPosition < i) {
					nodes[link.other].neighbors[link.indexPosition+1].indexPosition--;
				} else {
					nodes[link.other].neighbors[link.indexPosition].indexPosition--;
				}
			}
		}

		nodes[extNode].tensorObject->fix_mode(extNodeIndexPos, _slatePosition);
		nodes[extNode].neighbors.erase(nodes[extNode].neighbors.begin() + extNodeIndexPos);

		require_valid_network();

		contract_unconnected_subnetworks();
	}


	void TensorNetwork::remove_slate(const size_t _mode, const size_t _slatePosition) {
		require_valid_network();

		REQUIRE(_mode < order(), "invalid dimension to remove a slate from");
		REQUIRE(_slatePosition < dimensions[_mode], "invalide slate position to choose");
		REQUIRE(dimensions[_mode] > 0, "removing the last possible slate from this index position would result a dimension of size 0");

		const size_t extNode = externalLinks[_mode].other;
		const size_t extNodeIndexPos = externalLinks[_mode].indexPosition;

		externalLinks[_mode].dimension -= 1;
		dimensions[_mode] -= 1;
		if (nodes[extNode].tensorObject) {
			nodes[extNode].tensorObject->remove_slate(extNodeIndexPos, _slatePosition);
		}
		nodes[extNode].neighbors[extNodeIndexPos].dimension -= 1;
	}



	void TensorNetwork::resize_mode(const size_t _mode, const size_t _newDim, const size_t _cutPos) {
		REQUIRE(_mode < order(), "Invalid dimension given for resize_mode");
		require_valid_network();

		const size_t extNode = externalLinks[_mode].other;
		const size_t extNodeIndexPos = externalLinks[_mode].indexPosition;

		nodes[extNode].tensorObject->resize_mode(extNodeIndexPos, _newDim, _cutPos);
		nodes[extNode].neighbors[extNodeIndexPos].dimension = _newDim;
		externalLinks[_mode].dimension = _newDim;
		dimensions[_mode] = _newDim;

		require_valid_network();
	}


	void TensorNetwork::reduce_representation() {
		require_valid_network();

		TensorNetwork strippedNet = stripped_subnet();
		std::vector<std::set<size_t>> contractions(strippedNet.nodes.size());
		for (size_t id1=0; id1 < strippedNet.nodes.size(); ++id1) {
			TensorNode &currNode = strippedNet.nodes[id1];
			if (currNode.erased) {
				continue;
			}
			for (Link &l : currNode.neighbors) {
				if (l.external) { continue; }
				size_t r=1;
				for (Link &l2 : currNode.neighbors) {
					if (l2.other == l.other) {
						r *= l2.dimension;
					}
				}
				if (r*r >= currNode.size() || r*r >= strippedNet.nodes[l.other].size()) {
					if (contractions[id1].empty()) {
						contractions[id1].insert(id1);
					}
					if (contractions[l.other].empty()) {
						contractions[id1].insert(l.other);
					} else {
						contractions[id1].insert(contractions[l.other].begin(), contractions[l.other].end());
						contractions[l.other].clear();
					}
					strippedNet.contract(id1, l.other);
					id1 -= 1; // check the same node again in the next iteration
					break; // for-each iterator is broken, so we have to break
				}
			}
		}

		// perform the collected contractions from above
		for (std::set<size_t> &ids : contractions) {
			if (ids.size() > 1) {
				contract(ids);
			}
		}

		sanitize();
		require_valid_network();
	}


	void TensorNetwork::contract(const size_t _nodeId1, const size_t _nodeId2) {
		TensorNode &node1 = nodes[_nodeId1];
		TensorNode &node2 = nodes[_nodeId2];

		REQUIRE(!node1.erased, "It appears node1 = " << _nodeId1 << "  was already contracted?");
		REQUIRE(!node2.erased, "It appears node2 = " << _nodeId2 << "  was already contracted?");
		INTERNAL_CHECK(externalLinks.size() == order(), "Internal Error: " << externalLinks.size() << " != " << order());

		std::vector<TensorNetwork::Link> newLinks;
		newLinks.reserve(node1.order() + node2.order());

		if (!node1.tensorObject) {
			INTERNAL_CHECK(!node2.tensorObject, "Internal Error.");

			// Determine the links of the resulting tensor (first half)
			for ( const Link& l : node1.neighbors ) {
				if (!l.links(_nodeId1) && !l.links(_nodeId2)) {
					newLinks.emplace_back(l);
				}
			}
			// Determine the links of the resulting tensor (second half)
			for ( const Link& l : node2.neighbors ) {
				if (!l.links(_nodeId2) && !l.links(_nodeId1)) {
					newLinks.emplace_back(l);
				}
			}
		} else {
			INTERNAL_CHECK(node2.tensorObject, "Internal Error.");

			size_t contractedDimCount = 0;
			bool separated1;
			bool separated2;
			bool matchingOrder;

			// first pass of the links of node1 to determine
			//   1. the number of links between the two nodes,
			//   2. determine whether node1 is separated (ownlinks-commonlinks) or transposed separated (commonlinks-ownlinks)
			//   3. determine the links of the resulting tensor (first half)
			if(node1.order() > 1) {
				uint_fast8_t switches = 0;
				bool previous = node1.neighbors[0].links(_nodeId2);
				for (const Link& l : node1.neighbors) {
					if (l.links(_nodeId2)) {
						contractedDimCount++;
						if (!previous) {
							switches++;
							previous = true;
						}
					} else {
						newLinks.emplace_back(l);
						if (previous) {
							switches++;
							previous = false;
						}
					}
				}
				separated1 = (switches < 2);
			} else {
				if(!node1.neighbors.empty()) {
					if(node1.neighbors[0].links(_nodeId2)) {
						contractedDimCount = 1;
					} else {
						newLinks.emplace_back(node1.neighbors[0]);
					}
				}
				separated1 = true;
			}

			// first pass of the links of node2 to determine
			//   1. whether the order of common links is correct
			//   2. whether any self-links exist
			//   3. whether the second node is separated
			//   4. determine the links of the resulting tensor (second half)
			if(node2.order() > 1 && contractedDimCount > 0) {
				bool previous = node2.neighbors[0].links(_nodeId1);
				uint_fast8_t switches = 0;
				size_t lastPosOfCommon = 0;
				matchingOrder = true;
				for (const Link& l : node2.neighbors) {
					if (l.links(_nodeId1)) {
						if (l.indexPosition < lastPosOfCommon) {
							matchingOrder = false;
						}
						lastPosOfCommon = l.indexPosition;
						if (!previous) {
							switches++;
							previous = true;
						}
					} else {
						newLinks.emplace_back(l);
						if (previous) {
							switches++;
							previous = false;
						}
					}
				}
				separated2 = (switches < 2);
			} else {
				if(contractedDimCount == 0) {
					newLinks.insert(newLinks.end(), node2.neighbors.begin(), node2.neighbors.end());
				}

				separated2 = true;
				matchingOrder = true;
			}

			// Determine which (if any) node should be reshuffled
			// if order of common links does not match, reshuffle the smaller one
			if (!matchingOrder && separated1 && separated2) {
				if (node1.size() < node2.size()) {
					separated1 = false;
				} else {
					separated2 = false;
				}
			}

			// reshuffle first node
			if (!separated1) {
				std::vector<size_t> shuffle(node1.order());
				size_t pos = 0;

				for (size_t d = 0; d < node1.order(); ++d) {
					if (!node1.neighbors[d].links(_nodeId2)) {
						shuffle[d] = pos++;
					}
				}

				for (const Link& l : node2.neighbors) {
					if (l.links(_nodeId1)) {
						shuffle[l.indexPosition] = pos++;
					}
				}

				INTERNAL_CHECK(pos == node1.order(), "IE");
				reshuffle(*node1.tensorObject, *node1.tensorObject, shuffle);

				matchingOrder = true;
			}

			// reshuffle second node
			if (!separated2) {
				std::vector<size_t> shuffle(node2.order());
				size_t pos = 0;

				if (matchingOrder) {
					// Add common links in order as they appear in node2 to avoid both nodes changing to the opposite link order
					for (size_t d = 0; d < node2.order(); ++d) {
						if (node2.neighbors[d].links(_nodeId1)) {
							shuffle[d] = pos++;
						}
					}
				} else {
					for (const Link& l : node1.neighbors) {
						if (l.links(_nodeId2)) {
							shuffle[l.indexPosition] = pos++;
						}
					}
				}

				for (size_t d = 0; d < node2.order(); ++d) {
					if (!node2.neighbors[d].links(_nodeId1)) {
						shuffle[d] = pos++;
					}
				}

				INTERNAL_CHECK(pos == node2.order(), "IE");
				reshuffle(*node2.tensorObject, *node2.tensorObject, shuffle);
			}

			const bool trans1 = separated1 && !node1.neighbors.empty() && node1.neighbors[0].links(_nodeId2);
			const bool trans2 = separated2 &&!node2.neighbors.empty() &&!(node2.neighbors[0].links(_nodeId1));

			xerus::contract(*node1.tensorObject, *node1.tensorObject, trans1, *node2.tensorObject, trans2, contractedDimCount);
		}

		// Set Nodes
		nodes[_nodeId1].neighbors = std::move(newLinks);
		nodes[_nodeId2].erase();

		// Fix indices of other nodes // note that the indices that were previously part of node1 might also have changed
		for (size_t d = 0; d < nodes[_nodeId1].neighbors.size(); ++d) {
			const Link& l = nodes[_nodeId1].neighbors[d];
			if (l.external) {
				externalLinks[l.indexPosition].other = _nodeId1;
				externalLinks[l.indexPosition].indexPosition = d;
			} else {
				nodes[l.other].neighbors[l.indexPosition].other = _nodeId1;
				nodes[l.other].neighbors[l.indexPosition].indexPosition = d;
			}
		}

		require_valid_network(false);
	}


	double TensorNetwork::contraction_cost(const size_t _nodeId1, const size_t _nodeId2) const {
		REQUIRE(!nodes[_nodeId1].erased, "It appears node1 = " << _nodeId1 << " was already contracted?");
		REQUIRE(!nodes[_nodeId2].erased, "It appears node2 = " << _nodeId2 << " was already contracted?");

		if (_nodeId1 == _nodeId2) {
			return static_cast<double>(nodes[_nodeId1].size()); // Costs of a trace
		}

		//TODO add correct calculation for sparse matrices

		// Assume cost of mxr * rxn = m*n*r (which is a rough approximation of the actual cost for openBlas/Atlas)
		size_t cost = nodes[_nodeId1].size();
		for(const Link& neighbor : nodes[_nodeId2].neighbors) {
			if(!neighbor.links(_nodeId1)) {
				cost *= neighbor.dimension;
			}
		}
		return static_cast<double>(cost);
	}


	size_t TensorNetwork::contract(const std::set<size_t>& _ids) {
		// Trace out all single-node traces
		for ( const size_t id : _ids ) {
			perform_traces(id);
		}

		if (_ids.empty()) { return ~0ul; }

		if (_ids.size() == 1) { return *_ids.begin(); }

		if (_ids.size() == 2) {
			auto secItr = _ids.begin(); ++secItr;
			contract(*_ids.begin(), *secItr);
			return *_ids.begin();
		}

		if (_ids.size() == 3) {
			auto idItr = _ids.begin();
			const size_t a = *idItr; TensorNode &na = nodes[a]; ++idItr;
			const size_t b = *idItr; TensorNode &nb = nodes[b]; ++idItr;
			const size_t c = *idItr; TensorNode &nc = nodes[c];
			double sa  = 1, sb  = 1, sc  = 1; // sizes devided by the link dimensions between a,b,c
			double sab = 1, sbc = 1, sac = 1; // link dimensions
			for (size_t d = 0; d < na.order(); ++d) {
				if (na.neighbors[d].links(b)) {
					sab *= static_cast<double>(na.neighbors[d].dimension);
				} else if (na.neighbors[d].links(c)) {
					sac *= static_cast<double>(na.neighbors[d].dimension);
				} else {
					sa *= static_cast<double>(na.neighbors[d].dimension);
				}
			}
			for (size_t d = 0; d < nb.order(); ++d) {
				if (nb.neighbors[d].links(c)) {
					sbc *= static_cast<double>(nb.neighbors[d].dimension);
				} else if (!nb.neighbors[d].links(a)) {
					sb *= static_cast<double>(nb.neighbors[d].dimension);
				}
			}
			for (size_t d = 0; d < nc.order(); ++d) {
				if (!nc.neighbors[d].links(a) && !nc.neighbors[d].links(b)) {
					sc *= static_cast<double>(nc.neighbors[d].dimension);
				}
			}
			// cost of contraction a-b first etc.
			double costAB = sa*sb*sac*sbc*(sab+sc); // (sa*sac)*sab*(sb*sbc) + sa*sb*sac*sbc*sc;
			double costAC = sa*sc*sab*sbc*(sac+sb);
			double costBC = sb*sc*sab*sac*(sbc+sa);

			if (costAB < costAC && costAB < costBC) {
				LOG(TNContract, "contraction of ab first " << sa << " " << sb << " " << sc << " " << sab << " " << sbc << " " << sac);
				contract(a, b); contract(a, c);
			} else if (costAC < costBC) {
				LOG(TNContract, "contraction of ac first " << sa << " " << sb << " " << sc << " " << sab << " " << sbc << " " << sac);
				contract(a, c); contract(a, b);
			} else {
				LOG(TNContract, "contraction of bc first " << sa << " " << sb << " " << sc << " " << sab << " " << sbc << " " << sac);
				contract(b, c); contract(a, b);
			}
			return a;
		}


		TensorNetwork strippedNetwork = stripped_subnet([&](size_t _id){ return misc::contains(_ids, _id); });
		double bestCost = std::numeric_limits<double>::max();
		std::vector<std::pair<size_t, size_t>> bestOrder;

		// Ask the heuristics
		for (const internal::ContractionHeuristic &c : internal::contractionHeuristics) {
			c(bestCost, bestOrder, strippedNetwork);
		}

		INTERNAL_CHECK(bestCost < std::numeric_limits<double>::max() && !bestOrder.empty(), "Internal Error.");

		for (const std::pair<size_t,size_t> &c : bestOrder) {
			contract(c.first, c.second);
		}

		// Note: no sanitization as eg. TTStacks require the indices not to change after calling this function
		return bestOrder.back().first;
	}


	value_t TensorNetwork::frob_norm() const {
		const Index i;
		Tensor res;
		res() = (*this)(i&0) * (*this)(i&0);
		return std::sqrt(res[0]);
	}


	void TensorNetwork::draw(const std::string& _filename) const {
		std::stringstream graphLayout;

		graphLayout << "graph G {" << std::endl;
		graphLayout << "graph [mclimit=1000, maxiter=1000, overlap = false, splines = true]" << std::endl;

		for(size_t i = 0; i < nodes.size(); ++i) {
			// Create the Nodes
			if(nodes[i].erased) {
				graphLayout << "\tN"<<i<<" [label=\"N"<<i<<"\", shape=circle, fixedsize=shape, height=0.45];" << std::endl;
			} else {
				graphLayout << "\tN"<<i<<" [label=\"";
				for(size_t k=0; k+1 < nodes[i].order(); ++k) {
					if(nodes[i].order()/2 == k) {
						if(nodes[i].order()%2 == 0) {
						graphLayout << "<i"<<k<<"> "<<i<<"| ";
						} else {
							graphLayout << "<i"<<k<<"> N"<<i<<"| ";
						}
					} else if(nodes[i].order()%2 == 0 && nodes[i].order()/2 == k+1) {
						graphLayout << "<i"<<k<<"> N| ";
					} else {
						graphLayout << "<i"<<k<<"> | ";
					}
				}
				if(nodes[i].order() <= 2) {
					graphLayout << "<i"<<nodes[i].order()-1<<"> N"<<i<<"\", shape=record, fixedsize=shape, height=0.45, style=\"rounded,filled\"];" << std::endl;
				} else {
					graphLayout << "<i"<<nodes[i].order()-1<<">\", shape=record, fixedsize=shape, height=0.45, style=\"rounded,filled\"];" << std::endl;
				}

				// Add all links to nodes with smaller index and externals
				for(size_t j = 0; j < nodes[i].neighbors.size(); ++j) {
					if(nodes[i].neighbors[j].external) {
						graphLayout << "\t"<<nodes[i].neighbors[j].indexPosition<<" [shape=diamond, fixedsize=shape, height=0.38, width=0.38, style=filled];" << std::endl;
						graphLayout << "\tN"<<i<<":i"<<j<<" -- " << nodes[i].neighbors[j].indexPosition << " [len=1, label=\""<<nodes[i].neighbors[j].dimension<<"\"];" << std::endl;
					} else if(nodes[i].neighbors[j].other < i) {
						graphLayout << "\tN"<<i<<":i"<<j<<" -- " << "N"<<nodes[i].neighbors[j].other << ":i"<< nodes[i].neighbors[j].indexPosition<<" [label=\""<<nodes[i].neighbors[j].dimension<<"\"];" << std::endl;
					}
				}
			}
		}
		graphLayout << "}" << std::endl;
		misc::exec(std::string("dot -Tsvg > ") + _filename+".svg", graphLayout.str());
	}



	TensorNetwork operator*(const TensorNetwork &_lhs, const value_t _factor) {
		TensorNetwork res(*_lhs.get_copy());
		res *= _factor;
		return res;
	}

	TensorNetwork operator*(const value_t _factor, const TensorNetwork &_rhs) {
		TensorNetwork res(*_rhs.get_copy());
		res *= _factor;
		return res;
	}

	TensorNetwork operator/(const TensorNetwork &_lhs, const value_t _factor) {
		TensorNetwork res(*_lhs.get_copy());
		res *= 1.0/_factor;
		return res;
	}


	bool approx_equal(const TensorNetwork& _a, const TensorNetwork& _b, const value_t _eps) {
		REQUIRE(_a.dimensions == _b.dimensions, "The dimensions of the compared tensors don't match: " << _a.dimensions <<" vs. " << _b.dimensions);
		const Index i; // TODO no indices
		return frob_norm(_a(i&0) - _b(i&0)) <= _eps*(_a.frob_norm() + _b.frob_norm())/2.0;
	}


	bool approx_equal(const TensorNetwork& _a, const Tensor& _b, const value_t _eps) {
		return approx_equal(Tensor(_a), _b, _eps);
	}


	bool approx_equal(const Tensor& _a, const TensorNetwork& _b, const value_t _eps) {
		return approx_equal(_a, Tensor(_b), _eps);
	}

	namespace misc {

		void stream_writer(std::ostream &_stream, const TensorNetwork &_obj, const FileFormat _format) {
			if(_format == FileFormat::TSV) {
				_stream << std::setprecision(std::numeric_limits<value_t>::digits10 + 1);
			}
			// storage version number
			write_to_stream<size_t>(_stream, 1, _format);

			// Save dimensions
			write_to_stream(_stream, _obj.dimensions, _format);
			if(_format == FileFormat::TSV) { _stream << '\n'; }

			// save external links
			for(const TensorNetwork::Link& el : _obj.externalLinks) {
				write_to_stream<size_t>(_stream, el.other, _format);
				write_to_stream<size_t>(_stream, el.indexPosition, _format);
				write_to_stream<size_t>(_stream, el.dimension, _format);
			}
			if(_format == FileFormat::TSV) { _stream << "\n\n"; }

			// Save nodes with their links
			write_to_stream<size_t>(_stream, _obj.nodes.size(), _format);
			if(_format == FileFormat::TSV) { _stream << '\n'; }
			for(const TensorNetwork::TensorNode& node : _obj.nodes) {
				write_to_stream<size_t>(_stream, node.neighbors.size(), _format);
				for(const TensorNetwork::Link& link : node.neighbors) {
					write_to_stream<bool>(_stream, link.external, _format);
					write_to_stream<size_t>(_stream, link.other, _format);
					write_to_stream<size_t>(_stream, link.indexPosition, _format);
					write_to_stream<size_t>(_stream, link.dimension, _format);
				}
			}
			if(_format == FileFormat::TSV) { _stream << '\n'; }

			// Save tensorObjects
			for(const TensorNetwork::TensorNode& node : _obj.nodes) {
				write_to_stream(_stream, *node.tensorObject, _format);
				if(_format == FileFormat::TSV) { _stream << '\n'; }
			}
		}

		void stream_reader(std::istream& _stream, TensorNetwork &_obj, const FileFormat _format) {
			IF_CHECK( size_t ver = ) read_from_stream<size_t>(_stream, _format);
			REQUIRE(ver == 1, "Unknown stream version to open (" << ver << ")");

			// Load dimensions
			read_from_stream(_stream, _obj.dimensions, _format);

			// load external links
			_obj.externalLinks.resize(_obj.dimensions.size());
			for(TensorNetwork::Link& el : _obj.externalLinks) {
				el.external = false;
				el.other = read_from_stream<size_t>(_stream, _format);
				el.indexPosition = read_from_stream<size_t>(_stream, _format);
				el.dimension = read_from_stream<size_t>(_stream, _format);
			}

			// Load nodes with their links
			_obj.nodes.resize(read_from_stream<size_t>(_stream, _format));
			for(TensorNetwork::TensorNode& node : _obj.nodes) {
				node.neighbors.resize(read_from_stream<size_t>(_stream, _format));
				node.erased = false;
				for(TensorNetwork::Link& link : node.neighbors) {
					link.external = read_from_stream<bool>(_stream, _format);
					link.other = read_from_stream<size_t>(_stream, _format);
					link.indexPosition = read_from_stream<size_t>(_stream, _format);
					link.dimension = read_from_stream<size_t>(_stream, _format);
				}
			}

			// load tensorObjects
			for(TensorNetwork::TensorNode& node : _obj.nodes) {
				node.tensorObject.reset(new Tensor());
				read_from_stream<Tensor>(_stream, *node.tensorObject, _format);
			}

			_obj.require_valid_network();
		}
	} // namespace misc
} // namespace xerus
