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
 * @brief Header file for the TTBlock class.
 */

#pragma once

#include "tensor.h"
#include "tensorNetwork.h"
#include "ttNetwork.h"
#include "misc/check.h"

namespace xerus { namespace internal {

    class BlockTT;
    value_t move_block(BlockTT& _x, const size_t _position, const size_t _maxRank = ~0U);
    value_t move_core(BlockTT& _x, const size_t _position, const size_t _maxRank = ~0U);

    /**
     * @brief Specialized TensorNetwork class used to represent a BlockTT
     */
    class BlockTT final  {
    private:
        static const Index left, right, ext, p, r1, r2;

    public:
        size_t P;

        /**
         * @brief The position of the core.
         * @details CorePosition gives the position of the block/core tensor. All components
         * with smaller index are then left-orthogonalized and all components with larger index right-orthogonalized.
         */
        size_t corePosition;
        size_t blockPosition;

        std::vector<Tensor> components;
        std::vector<size_t> dimensions;


        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
        /**
         * @brief BlockTTs can be default construced.
         */
        BlockTT() = default;

        ///@brief Constructs an zero initialized TTNetwork with the given dimensions, ranks and block dimensions.
        BlockTT(const std::vector<size_t>& _dimensions, const std::vector<size_t>& _ranks, const size_t _blockPosition, const size_t _blockDim);


        ///@brief BlockTTs are default copy constructable.
        BlockTT(const BlockTT & _cpy) = default;


        ///@brief BlockTTs are default move constructable.
        BlockTT(      BlockTT&& _mov) noexcept = default;


        /**
         * @brief Constructs a BlockTT from the given TTTensor.
         */
        explicit BlockTT(const TTTensor& _tttensor, const size_t _blockPosition, const size_t _blockDim);



        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Standard Operators - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
        ///@brief BlockTTs are default assignable.
        BlockTT& operator=(const BlockTT&  _other) = default;


        ///@brief BlockTTs are default move-assignable.
        BlockTT& operator=(      BlockTT&& _other) = default;


        /*- - - - - - - - - - - - - - - - - - - - - - - - - - Miscellaneous - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
        public:

            size_t degree() const;


            Tensor& component(const size_t _idx);


            const Tensor& get_component(const size_t _idx) const;


            void set_component(const size_t _idx, const Tensor& _T);


            Tensor get_core(const size_t _blockPos) const;


            Tensor get_average_core() const;


            TTTensor get_average_tt() const;


            /**
            * @brief Gets the ranks of the TTNetwork.
            * @return A vector containing the current ranks.
            */
            std::vector<size_t> ranks() const;

            ///@brief Return the number of components
            size_t num_components() const;


            /**
            * @brief Gets the rank of a specific egde of the TTNetwork.
            * @param _i Position of the edge in question.
            * @return The current rank of edge _i.
            */
            size_t rank(const size_t _i) const;


            void move_core_left(const double _eps, const size_t _maxRank);

            void move_core_right(const double _eps, const size_t _maxRank);


            /**
            * @brief Move the core to a new position.
            */
            void move_core(const size_t _position, const double _eps=EPSILON, const size_t _maxRank=std::numeric_limits<size_t>::max());
            value_t move_core(const size_t _position, const size_t _maxRank);

            template<class distribution=std::normal_distribution<value_t>, class generator=std::mt19937_64>
            static BlockTT random(const std::vector<size_t> _dimensions, const std::vector<size_t> &_ranks, const size_t _blockPosition, const size_t _blockDim, distribution& _dist=xerus::misc::defaultNormalDistribution, generator& _rnd=xerus::misc::randomEngine) {
            /* template<class distribution, class generator> */
            /* BlockTT BlockTT::random(const std::vector<size_t> _dimensions, const std::vector<size_t> &_ranks, const size_t _blockPosition, const size_t _blockDim, distribution& _dist, generator& _rnd) { */
                const size_t numComponents = _dimensions.size();
                XERUS_REQUIRE(_ranks.size()+1 == numComponents,"Non-matching amount of ranks given to BlockTT::random.");
                XERUS_REQUIRE(!misc::contains(_dimensions, size_t(0)), "Trying to construct a BlockTT-Tensor with dimension 0 is not possible.");
                XERUS_REQUIRE(!misc::contains(_ranks, size_t(0)), "Trying to construct random BlockTT-Tensor with rank 0 is illegal.");
                XERUS_REQUIRE(_blockPosition < _dimensions.size(), "_blockPosition >= _dimensions.size()");

                BlockTT result(_dimensions, _ranks, _blockPosition, _blockDim);
                result.corePosition=0;

                for(size_t i = 0; i < numComponents; ++i) {
                    std::vector<size_t> cmpDims;
                    if (i == _blockPosition) {
                        cmpDims.reserve(4);
                        cmpDims.push_back((i>0) ? _ranks[i-1] : 1);
                        cmpDims.push_back(_dimensions[i]);
                        cmpDims.push_back(_blockDim);
                        cmpDims.push_back((i<numComponents-1) ? _ranks[i] : 1);
                    }
                    else {
                        cmpDims.reserve(3);
                        cmpDims.push_back((i>0) ? _ranks[i-1] : 1);
                        cmpDims.push_back(_dimensions[i]);
                        cmpDims.push_back((i<numComponents-1) ? _ranks[i] : 1);
                    }

                    const auto rndComp = Tensor::random(cmpDims, _dist, _rnd);
                    result.set_component(i, rndComp);
                    if (i < numComponents-1) xerus::internal::move_core(result, i+1);
                    /* result.components[i] = rndComp; */
                }

                /* result.corePosition = _dimensions.size()-1; */
                /* result.move_core(0); */
                xerus::internal::move_core(result, _blockPosition);
                return result;
            }

            /* template<class distribution=std::normal_distribution<value_t>, class generator=std::mt19937_64> */
            /* BlockTT random(const std::vector<size_t>& _dimensions, const size_t _rank, distribution& _dist, generator& _rnd) { */
            /*     return TTNetwork::random(_dimensions, std::vector<size_t>(_dimensions.size()-1, _rank), _dist, _rnd); */
            /* } */

            void average_core();

            bool all_entries_valid() const;

            value_t frob_norm() const;

            size_t dofs() const;

    };

    void stream_writer(std::ostream& _stream, const BlockTT &_obj, misc::FileFormat _format);
    void stream_reader(std::istream& _stream, BlockTT &_obj, const misc::FileFormat _format);
    value_t frob_norm(const BlockTT& _x);
} }
