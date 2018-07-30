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

namespace xerus { namespace internal {
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
            
            
            void average_core();

            bool all_entries_valid() const;

            value_t frob_norm() const;

            size_t dofs() const;

    };

    void stream_writer(std::ostream& _stream, const BlockTT &_obj, misc::FileFormat _format);
    void stream_reader(std::istream& _stream, BlockTT &_obj, const misc::FileFormat _format);
    value_t frob_norm(const BlockTT& _x);
} }
