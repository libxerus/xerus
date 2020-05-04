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
* @brief Header file for the TensorNetwork class.
*/

#pragma once

#include <set>
#include <memory>
#include <functional>
#include "misc/fileIO.h"

#include "indexedTensor.h"

namespace xerus {
    // Necessary forward declaritons
    class Tensor;

    /**
    * @brief Very general class used to represent arbitary tensor networks.
    * @details Used as a basis for tensor decompositions like the TTNetwork but also used for the lazy evaluation of Tensor contractions.
    */
    class TensorNetwork {
    public:
        ///@brief: Represention of the ranks of a TensorNetwork.
        using RankTuple = std::vector<size_t>;

        /**
        * @brief Class representing a link from a TensorNode to another node or an external index.
        */
        class Link final {
        public:
            ///@brief The index of the otherNode this Link links to.
            size_t other;

            ///@brief IndexPosition on the other node or index of external index.
            size_t indexPosition;

            ///@brief Dimension of the link, always equals to other->tensorObject->dimensions[indexPosition].
            size_t dimension;

            ///@brief Flag indicating whether this link correspond to an external index.
            bool external;

            Link() noexcept = default;
            Link(const Link& ) noexcept = default;
            Link(      Link&&) noexcept = default;

            Link(const size_t _other, const size_t _indexPos, const size_t _dim, const bool _external) noexcept;

            Link& operator=(const Link& ) noexcept = default;
            Link& operator=(      Link&&) noexcept = default;

            /**
            * @brief Checks whether this link links to a particular node
            * @param _other the other node for which the linkage shall be checked
            * @return TRUE if _other is the target of this Link, FALSE otherwise.
            */
            bool links(const size_t _other) const noexcept;
        };


        /**
        * @brief The TensorNode class is used by the class TensorNetwork to store the componentent tensors defining the network.
        */
        class TensorNode final {
        public:
            ///@brief Save slot for the tensorObject associated with this node.
            std::unique_ptr<Tensor> tensorObject;

            ///@brief Vector of links defining the connection of this node to the network.
            std::vector<Link> neighbors;

            ///@brief Internal Flag
            bool erased;

            explicit TensorNode();

            TensorNode(const TensorNode&  _other);
            TensorNode(      TensorNode&& _other) noexcept = default;

            explicit TensorNode(      std::unique_ptr<Tensor>&& _tensorObject);

            explicit TensorNode(std::unique_ptr<Tensor>&& _tensorObject, std::vector<Link> _neighbors);

            ~TensorNode();

            TensorNode& operator=(const TensorNode&  _other);
            TensorNode& operator=(      TensorNode&& _other) noexcept;

            TensorNode strippped_copy() const;

            // All getters are written without the use of tensorObject so that they also work for empty nodes

            size_t size() const noexcept;

            size_t order() const noexcept;

            void erase() noexcept;
        };

    protected:

        /**
         * @brief Internal indicator to prevent the creation of an order zero node in TensorNetwork constructor.
         */
        enum class ZeroNode : bool { None, Add };


    public:
        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Member variables - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

        ///@brief Dimensions of the external indices, i.e. the dimensions of the tensor represented by the network.
        std::vector<size_t> dimensions;

        ///@brief The nodes constituting the network. The order determines the ids of the nodes.
        std::vector<TensorNode> nodes;

        ///@brief The open links of the network in order.
        std::vector<Link> externalLinks;

        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

        /**
        * @brief Constructs an order zero TensorNetwork.
        * @details The order of an empty TN is zero.
        */
        explicit TensorNetwork();


        ///@brief Copy Constructor
        TensorNetwork(const TensorNetwork& _cpy) = default;


        ///@brief Move Constructor
        TensorNetwork(TensorNetwork&& _mv) noexcept = default;


        /**
        * @brief Constructs the trivial TensorNetwork containing the given Tensor as single node.
        */
        TensorNetwork(Tensor _other);


        /**
        * @brief Constructs the trivial TensorNetwork containing the given Tensor as single node.
        * The TN takes the ownership of the pointer.
        */
        TensorNetwork(std::unique_ptr<Tensor>&&  _tensor);


        /**
        * @brief Constructs the trivial TensorNetwork containing a Tensor with the given order.
        * @details All dimensions are set equals one and the only entry
        * of the tensor is zero.
        */
        TensorNetwork(size_t _order);

        /**
         * @brief (Internal) Constructs an order zero TensorNetwork.
         * @details The order of an empty TN is zero.
         * @param _nodeStatus defines whether the network will contain one order zero node with the single
         * entry zero.
         */
        explicit TensorNetwork(const ZeroNode _nodeStatus);


        ///@brief Destructor
        virtual ~TensorNetwork() = default;


        /**
        * @brief Returns a new copy of the network.
        */
        virtual TensorNetwork* get_copy() const;

    protected:
        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Internal Helper functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
        ///@brief: Sets the externalLinks and returns an Link vector for a node, assuming that this node is the only node there is and all given dimensions are open.
        std::vector<Link> init_from_dimension_array();


        /**
         * @brief Creates a dataless copy of a subnet.
         * @details Creates a copy of this TensorNetwork containing the specified nodes,
         * but does not propagate the data. Instead it uses the nullptr as data for all nodes.
         * @param _idF a function returning true if its argument should be part of the stripped subnet. defaults to selecting all nodes.
         * @return the new TensorNetwork.
         */
        TensorNetwork stripped_subnet(const std::function<bool(size_t)>& _idF = [](size_t){ return true;}) const;


        /**
        * @brief Contracts all nodes that are not connected to any external links.
        */
        virtual void contract_unconnected_subnetworks();


        /**
         * @brief Performs all traces in the given node.
         * @param _nodeId id of the node for which the traces shall be performed.
         */
        void perform_traces(const size_t _nodeId);

    public:

        /**
         * @brief Finds the position of a single common edge between two nodes.
         * @param _nodeA The first node.
         * @param _nodeB The second node.
         * @return Tuple containing the two positions in @a _nodeA and @a _nodeB.
         */
        std::pair<size_t, size_t> find_common_edge(const size_t _nodeA, const size_t _nodeB) const;



        /**
         * @brief Removes all erased nodes from the TensorNetwork.
         * @details The order of the node ids is retained, but the ids might decrease due to removed nodes.
         */
        void sanitize();


        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Standard operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

        ///@brief TensorNetworks are copy assignable.
        TensorNetwork& operator=(const TensorNetwork &_cpy) = default;

        ///@brief TensorNetworks are move assignable.
        TensorNetwork& operator=(TensorNetwork &&_mv) = default;


        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Conversions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

        /**
        * @brief Explicit cast to Tensor.
        * @details Contracts the complete network into a single Tensor
        */
        explicit operator Tensor() const;

        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Access - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
        /**
        * @brief Read the value at a specific position.
        * @details This allows the efficent calculation of a single entry of the TensorNetwork, by first fixing the external dimensions
        * and then completly contracting the network. Do NOT use this as a manual cast to Tensor (there is an explicit cast for that).
        * @param _position the position of the entry to be read assuming row-major ordering and a single node.
        * @returns the calculated value (NO reference)
        */
        value_t operator[](const size_t _position) const;


        /**
        * @brief Read the value at a specific position.
        * @details This allows the efficent calculation of a single entry of the TensorNetwork, by first fixing the external dimensions
        * and then completly contracting the network. Do NOT use this as a manual cast to Tensor (there is an explicit cast for that).
        * @param _positions the position of the entry to be read assuming a single node.
        * @returns the calculated value (NO reference)
        */
        value_t operator[](const std::vector<size_t>& _positions) const;


        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Basic arithmetics - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

        /**
        * @brief Performs the entrywise multiplication with a constant @a _factor.
        * @details Internally this only results in a change in the global factor.
        * @param _factor the factor,
        */
        virtual void operator*=(const value_t _factor);


        /**
        * @brief Performs the entrywise divison by a constant @a _divisor.
        * @details Internally this only results in a change in the global factor.
        * @param _divisor the divisor,
        */
        virtual void operator/=(const value_t _divisor);


        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Indexing - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
        /**
        * @brief Indexes the TensorNetwork for read/write use.
        * @param _args several [indices](@ref Index) determining the desired index order.
        * @return an internal representation of an IndexedTensor(Network).
        */
        template<typename... args>
        internal::IndexedTensor<TensorNetwork> operator()(args... _args) {
            return internal::IndexedTensor<TensorNetwork>(this, std::vector<Index>({_args...}), false);
        }


        /**
        * @brief Indexes the TensorNetwork for read only use.
        * @param _args several [indices](@ref Index) determining the desired index order.
        * @return an internal representation of an IndexedTensor(Network).
        */
        template<typename... args>
        internal::IndexedTensorReadOnly<TensorNetwork> operator()(args... _args) const {
            return internal::IndexedTensorReadOnly<TensorNetwork>(this, std::vector<Index>({_args...}));
        }


        /**
        * @brief Indexes the TensorNetwork for read/write use.
        * @param _indices several [indices](@ref Index) determining the desired index order.
        * @return an internal representation of an IndexedTensor(Network).
        */
        internal::IndexedTensor<TensorNetwork> operator()(const std::vector<Index> & _indices);


        /**
        * @brief Indexes the TensorNetwork for read/write use.
        * @param _indices several [indices](@ref Index) determining the desired index order.
        * @return an internal representation of an IndexedTensor(Network).
        */
        internal::IndexedTensor<TensorNetwork> operator()(std::vector<Index>&& _indices);


        /**
        * @brief Indexes the TensorNetwork for read only use.
        * @param _indices several [indices](@ref Index) determining the desired index order.
        * @return an internal representation of an IndexedTensor(Network).
        */
        internal::IndexedTensorReadOnly<TensorNetwork> operator()(const std::vector<Index> & _indices) const;


        /**
        * @brief Indexes the TensorNetwork for read only use.
        * @param _indices several [indices](@ref Index) determining the desired index order.
        * @return an internal representation of an IndexedTensor(Network).
        */
        internal::IndexedTensorReadOnly<TensorNetwork> operator()(std::vector<Index>&& _indices) const;


        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Operator specializations - - - - - - - - - - - - - - - - - - - - - - - - - - */

        ///@brief (Internal) Calculates the contraction between _me and _other and stores the result in _out. Requires that *this is the tensorObjectReadOnly of _me.
        virtual bool specialized_contraction(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& _out, internal::IndexedTensorReadOnly<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) const;


        ///@brief (Internal) Calculates the sum between _me and _other and stores the result in _out. Requires that *this is the tensorObjectReadOnly of _me.
        virtual bool specialized_sum(std::unique_ptr<internal::IndexedTensorMoveable<TensorNetwork>>& _out, internal::IndexedTensorReadOnly<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other) const;


        ///@brief (Internal) Evaluates _other into _me. Requires that *this is the tensorObjectReadOnly of _me.
        virtual void specialized_evaluation(internal::IndexedTensorWritable<TensorNetwork>&& _me, internal::IndexedTensorReadOnly<TensorNetwork>&& _other);

        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

        /**
        * @brief [Deprecated] Gets the degree of the TensorNetwork.
        * @details The degree is defined as the number of dimensions (i.e. dimensions.size())
        * and is always equal to the number of externalLinks (i.e. externalLinks.size()).
        * @return the degree.
        */
		XERUS_deprecated("TensorNetwork::degree() is deprecated and will be removed in a future version. Use TensorNetwork::order() instead.")
        size_t degree() const;
		
		/**
        * @brief Gets the order of the TensorNetwork.
        * @details The order is defined as the number of dimensions (i.e. dimensions.size())
        * and is always equal to the number of externalLinks (i.e. externalLinks.size()).
        * @return the order.
        */
        size_t order() const;

        /**
        * @brief Calculates the storage requirement of the current representation.
        * @return The datasize in sizeof(value_t).
        */
        size_t datasize() const;


        /// @brief reshuffled the nodes according to the given function
        void reshuffle_nodes(const std::function<size_t(size_t)>& _f);


        /**
        * @brief Sanity checks the network.
        * @details Checks whether all links in the network are set consistently and matching the
        * underlying tensor objects. If not an exception is throws and the function does not return.
        * Note that this only checks whether the TensorNetwork is valid not whether the additional
        * constrains of a specific format are fullfilled. For this purpose use is_in_expected_format().
        */
        void require_valid_network(const bool _check_erased = true) const;


        /**
         * @brief Sanity check for the TensorNetwork and if applicable for the specific format.
         * @details Checks whether all links in the network are set consistently and matching the
         * underlying tensor objects. This also checks whether the additional constrains of the specific
         * format (if any) are fullfilled.
         */
        virtual void require_correct_format() const;


        /**
        * @brief Swaps the external indices @a _i and @a _j, effectively changing those indices for the
        * represented Tensor (e.g. a transposition for matrices).
        */
        void swap_external_links(const size_t _i, const size_t _j);


        /**
        * @brief Inserts all nodes from @a _toInsert into @a _base, creating links where demanded by the indices.
        */
        static void add_network_to_network(internal::IndexedTensorWritable<TensorNetwork>&& _base, internal::IndexedTensorReadOnly<TensorNetwork>&& _toInsert);


        /**
         * @brief Finds traces defined by the indices and internally links the corresponding indices. Also applys all fixed indices
         * @details For each trace this reduces the order of the TN by two and removes two indices from the IndexedTensor.
         */
        static void link_traces_and_fix(internal::IndexedTensorWritable<TensorNetwork>&& _base);



    public:
        /**
        * @brief Thresholds the rank between two given nodes.
        * @details The given nodes must be joined by a single edge. Both nodes are contracted and an SVD is calculated to perform the thresholding.
        * The obtained core is contracted to nodeB, i.e. nodeA remains orthogonal in the used matrification.
        * @param _nodeA First node that takes part in the rank thresholding. This node remains orthogonalized.
        * @param _nodeB Second node that takes part in the rank thresholding. This nodes carries the core afterwards.
        * @param _maxRank Maximal allowed rank.
        * @param _eps Epsilion to be used in the SVD to determine zero singular values.
        * @param _softThreshold Softthreshold that is to be applied.
        */
        virtual void round_edge(const size_t _nodeA, const size_t _nodeB, const size_t _maxRank, const double _eps, const double _softThreshold);


        /**
        * @brief Transfers the core from one given node to another.
        * @details The given nodes must be joined by a single edge. A QR decomposition of the first node is calculated and the core contracted to the second one.
        * @param _from First node, which remains orthogonalized.
        * @param _to Second node, which carries the core afterwards.
        * @param _allowRankReduction Flag indicating whether a rank revealing decomposition is to be used which allows the reduction of the rank.
        */
        virtual void transfer_core(const size_t _from, const size_t _to, const bool _allowRankReduction = true);


        /**
        * @brief Contracts all nodes that are joined by a full-rank edge.
        * @details This reduces the overall storage requirements and can be useful to store intermediate results e.g. after fixing one of several indices.
        */
        void reduce_representation();


        /**
        * @brief Fixes a specific mode to a specific value, effectively reducing the order by one.
        * @param _mode the mode in which the slate shall be fixed, e.g. 0 to fix the first mode.
        * @param _slatePosition the position in the corresponding mode that shall be used.
        */
        virtual void fix_mode(const size_t _mode, const size_t _slatePosition);

        /**
         * @brief removes the given @a _slatePosition from the @a _mode. this reduces the given dimension by one
         */
        virtual void remove_slate(const size_t _mode, const size_t _slatePosition);

        /**
         * @brief Resizes a specific mode of the TensorNetwork.
         * @param _mode the mode to resize.
         * @param _newDim the new value that resized mode shall have.
         * @param _cutPos the position within the selected mode in front of which slates are inserted
         * or removed. By default the current dimension, i.e new slates are added after the last current one
         * and removed starting from the last one.
         */
        virtual void resize_mode(const size_t _mode, const size_t _newDim, const size_t _cutPos=~0ul);

        /**
         * @brief Contracts the nodes with indices @a _nodeId1 and @a _nodeId2.
         * @details Replaces @a _nodeId1 with the contraction and erases @a _nodeId2.
         * @param _nodeId1 The first node, that will contain the result afterwards.
         * @param _nodeId2 The second node, that will be erased afterwards.
        */
        void contract(const size_t _nodeId1, const size_t _nodeId2);


        /**
        * @brief Approximates the cost of contraction two given nodes.
        * @param _nodeId1 id of the first node.
        * @param _nodeId2 id of the second node.
        * @return The approxiamted contraction cost.
        */
        double contraction_cost(const size_t _nodeId1, const size_t _nodeId2) const;


        /**
         * @brief Contracts the nodes with with indices included in the given set @a _ids.
         * @details Erases all nodes but one, which id is returned.
         * @param _ids set with all ids to be erased.
         * @return The id in which the result of the contraction is stored.
         */
        size_t contract(const std::set<size_t>& _ids);


        /**
        * @brief Calculates the frobenious norm of the TensorNetwork.
        * @return the frobenious norm of the TensorNetwork.
        */
        virtual value_t frob_norm() const;


        /**
        * @brief Draws a graph representation of the TensorNetwork.
        * @details The drawing is realized by a system call to "dot" which plots the graph structure.
        * @param _filename path and name of the file where to save the image.
        */
        void draw(const std::string& _filename) const;
    };

    TensorNetwork operator*(const TensorNetwork &_lhs, const value_t _factor);

    TensorNetwork operator*(const value_t _factor, const TensorNetwork &_rhs);

    TensorNetwork operator/(const TensorNetwork &_lhs, const value_t _factor);

    /**
     * @brief Calculates the frobenious norm of the given TensorNetwork.
     * @param _network the TensorNetwork of which the frobenious norm shall be calculated.
     * @return the frobenious norm.
     */
    static XERUS_force_inline value_t frob_norm(const TensorNetwork& _network) { return _network.frob_norm(); }


    /**
     * @brief Checks whether two TensorNetworks are approximately equal.
     * @details Check whether ||@a _a - @a _b ||/(||@a a ||/2 + ||@a _b ||/2) < @a _eps, i.e. whether the relative difference in the frobenius norm is sufficently small.
     * @param _a the first test candidate.
     * @param _b the second test candidate
     * @param _eps the maximal relative difference between @a _a and @a _b.
     * @return TRUE if @a _a and @a _b are determined to be approximately equal, FALSE otherwise.
     */
    bool approx_equal(const TensorNetwork& _a, const TensorNetwork& _b, const value_t _eps = EPSILON);


    /**
     * @brief Convinience wrapper, casts the the given TensorNetwork @a _a to Tensor and calls the Tensor function.
     */
    bool approx_equal(const TensorNetwork& _a, const Tensor& _b, const value_t _eps = EPSILON);


    /**
     * @brief Convinience wrapper, casts the the given TensorNetwork @a _b to Tensor and calls the Tensor function.
     */
    bool approx_equal(const Tensor& _a, const TensorNetwork& _b, const value_t _eps = EPSILON);


    std::ostream &operator<<(std::ostream &_out, const TensorNetwork::Link &_rhs);

    namespace misc {
        /**
        * @brief Pipes all information necessary to restore the current TensorNetwork into @a _stream.
        * @note that this excludes header information
        */
        void stream_writer(std::ostream &_stream, const TensorNetwork &_obj, const FileFormat _format);

        /**
        * @brief Restores the TensorNetwork from a stream of data.
        */
        void stream_reader(std::istream &_stream, TensorNetwork &_obj, const FileFormat _format);
    }
}
