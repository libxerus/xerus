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
* @brief Implementation of the TTNetwork class (and thus TTTensor and TTOperator).
*/

#include <algorithm>
#include <memory>

#include <xerus/blockTT.h>
#include <xerus/ttNetwork.h>

#include <xerus/misc/check.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>

#include <xerus/basic.h>
#include <xerus/misc/basicArraySupport.h>

namespace xerus { namespace internal {
    const Index BlockTT::left, BlockTT::right, BlockTT::ext, BlockTT::p, BlockTT::r1, BlockTT::r2;
    /*- - - - - - - - - - - - - - - - - - - - - - - - - - Constructors - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/

    BlockTT::BlockTT(const std::vector<size_t>& _dimensions, const std::vector<size_t>& _ranks, const size_t _blockPosition, const size_t _blockDim) :
        P(_blockDim),
        corePosition(_blockPosition),
        blockPosition(_blockPosition),
        dimensions(_dimensions)
    {
        REQUIRE(_dimensions.size() == _ranks.size()+1, "_dimensions.size() != _ranks.size()+1");
        REQUIRE(_blockPosition < _dimensions.size(), "_blockPosition >= _dimensions.size()");

        const auto numComponents = _dimensions.size();
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

            components.emplace_back(cmpDims);
        }
    }

    BlockTT::BlockTT(const TTTensor& _tttensor, const size_t _blockPosition, const size_t _blockDim) :
        P(_blockDim),
        corePosition(_blockPosition),
        blockPosition(_blockPosition),
        dimensions(_tttensor.dimensions)
    {
        REQUIRE(_tttensor.canonicalized && _tttensor.corePosition == _blockPosition, "Invalid core Position");
        _tttensor.require_correct_format();

        // Get components
        for(size_t i = 0; i < _tttensor.degree(); ++i) {
            components.push_back(_tttensor.get_component(i));
        }

        // Create block
        components[_blockPosition](left, ext, p, right) = components[_blockPosition](left, ext, right)*Tensor::ones({_blockDim})(p);
    }

    size_t BlockTT::degree() const {
        return components.size();
    }

    std::vector<size_t> BlockTT::ranks() const {
        std::vector<size_t> res;
        res.reserve(degree()-1);
        for (size_t n = 1; n < degree(); ++n) {
            res.push_back(components[n].dimensions[0]);
        }
        return res;
    }

    size_t BlockTT::num_components() const { return components.size(); }

    size_t BlockTT::rank(const size_t _idx) const {
        REQUIRE(_idx+1 < degree(), "Illegal index " << _idx <<" in TTNetwork::component, as there are onyl " << degree() << " components.");
        return components[_idx+1].dimensions[0];
    }


    Tensor& BlockTT::component(const size_t _idx) {
        REQUIRE(_idx < degree(), "Illegal index " << _idx <<" in TTNetwork::component, as there are onyl " << degree() << " components.");
        return components[_idx];
    }


    const Tensor& BlockTT::get_component(const size_t _idx) const {
        REQUIRE(_idx < degree(), "Illegal index " << _idx <<" in TTNetwork::component, as there are onyl " << degree() << " components.");
        return components[_idx];
    }


    void BlockTT::set_component(const size_t _idx, const Tensor& _T) {
        REQUIRE(_idx == corePosition, "Illegal index " << _idx <<" as core position is " << corePosition << ".");
        REQUIRE(components[_idx].dimensions == _T.dimensions, "Invalid dimensions given: " << components[_idx].dimensions << " vs " << _T.dimensions);
        components[_idx] = _T;
    }


    Tensor BlockTT::get_core(const size_t _blockPos) const {
        REQUIRE(_blockPos < P, "IE");
        Tensor coreCmp = components[corePosition];
        coreCmp.fix_mode(2, _blockPos);
        return coreCmp;
    }


    Tensor BlockTT::get_average_core() const {
        Tensor coreCmp;
        coreCmp(left, ext, right) = (1.0/double(P))*components[corePosition](left, ext, p, right)*Tensor::ones({P})(p);
        return coreCmp;
    }


    TTTensor BlockTT::get_average_tt() const {
        TTTensor ttRep(dimensions);

        for(size_t i = 0; i < degree(); i++) {
            if(i == corePosition) {
                ttRep.set_component(i, get_average_core());
            } else {
                ttRep.set_component(i, get_component(i));
            }
        }

        return ttRep;
    }


    value_t BlockTT::frob_norm() const {
        return components[corePosition].frob_norm();
    }
    value_t frob_norm(const BlockTT& _x) { return _x.frob_norm(); }


    size_t BlockTT::dofs() const {
        size_t numDofs = 0;

        for(const auto& comp : components) {
            numDofs += comp.size;
        }

        for(const auto r : ranks()) {
            numDofs -= r*r;
        }


        REQUIRE(corePosition == 0, "IE");
        numDofs -= components[0].size - misc::sqr(components[1].dimensions[0]);

        return numDofs;
    }


    void BlockTT::move_core_left(const double _eps, const size_t _maxRank) {
        REQUIRE(corePosition > 0, "Can't move core left from position " << corePosition);

        if(P == 1) {
            Tensor Q, R;

            (R(left, p, r1), Q(r1, ext, right)) = RQ(components[corePosition](left, ext, p, right));

            components[corePosition] = Q;
            components[corePosition-1](left, ext, p, right) = components[corePosition-1](left, ext, r1)*R(r1, p, right);
        } else {
            Tensor U, S, V;

            (U(left, p, r1), S(r1, r2), V(r2, ext, right)) = SVD(components[corePosition](left, ext, p, right), _maxRank, _eps);
            components[corePosition] = V;
            components[corePosition-1](left, ext, p, right) = components[corePosition-1](left, ext, r1)*U(r1, p, r2)*S(r2, right);
        }
        corePosition--;
        blockPosition--;
    }


    void BlockTT::move_core_right(const double _eps, const size_t _maxRank) {
        REQUIRE(corePosition+1 < degree(), "Can't move core right from position " << corePosition);

        if(P == 1) {
            Tensor Q, R;

            (Q(left, ext, r1), R(r1, p, right)) = QR(components[corePosition](left, ext, p, right));
            components[corePosition] = Q;
            components[corePosition+1](left, ext, p, right) = R(left, p, r1)*components[corePosition+1](r1, ext, right);
        } else {
            Tensor U, S, V;

            (U(left, ext, r1), S(r1, r2), V(r2, p, right)) = SVD(components[corePosition](left, ext, p, right), _maxRank, _eps);
            components[corePosition] = U;
            components[corePosition+1](left, ext, p, right) = S(left, r1)*V(r1, p, r2)*components[corePosition+1](r2, ext, right);
        }
        corePosition++;
        blockPosition++;
    }


    void BlockTT::move_core(const size_t _position, const double _eps, const size_t _maxRank) {
        REQUIRE(_position < degree(), "Invalid new core position " << _position);

        while(corePosition < _position) {
            move_core_right(_eps, _maxRank);
        }

        while(corePosition > _position) {
            move_core_left(_eps, _maxRank);
        }
    }

    value_t BlockTT::move_core(const size_t _position, const size_t _maxRank) {
        REQUIRE(_maxRank > 0, "maxRank must be larger than zero.");
        REQUIRE(_position < degree(), "IE");
        Tensor U, S, V;
        value_t ret = 0; // corePosition == _position

        while(corePosition < _position) { // To right
            const Tensor& X = components[corePosition];
            ret = calculate_svd(U, S, V, X, 2, _maxRank, .0);

            components[corePosition] = U;
            components[corePosition+1](left, ext, p, right) = S(left, r1)*V(r1, p, r2)*components[corePosition+1](r2, ext, right);
            corePosition++;
            blockPosition++;
        }

        while(corePosition > _position) { // To left
            const Tensor X = reshuffle(components[corePosition], {0,2,1,3});
            ret = calculate_svd(U, S, V, X, 2, _maxRank, .0);

            components[corePosition] = V;
            components[corePosition-1](left, ext, p, right) = components[corePosition-1](left, ext, r1)*U(r1, p, r2)*S(r2, right);
            corePosition--;
            blockPosition--;
        }

        return ret;
    }

    void BlockTT::average_core() {
        Tensor coreCmp;
        coreCmp(left, ext, right) = (1.0/double(P))*components[corePosition](left, ext, p, right)*Tensor::ones({P})(p);
        components[corePosition](left, ext, p, right) = coreCmp(left, ext, right)*Tensor::ones({P})(p);
    }


    void stream_writer(std::ostream& _stream, const BlockTT &_obj, misc::FileFormat _format) {
        if(_format == misc::FileFormat::TSV) {
            _stream << std::setprecision(std::numeric_limits<value_t>::digits10 + 1);
        }
        // storage version number
        xerus::misc::write_to_stream<size_t>(_stream, 1, _format);

        xerus::misc::write_to_stream<size_t>(_stream, _obj.P, _format);
        xerus::misc::write_to_stream<size_t>(_stream, _obj.corePosition, _format);
        xerus::misc::write_to_stream<size_t>(_stream, _obj.blockPosition, _format);
        xerus::misc::write_to_stream<std::vector<size_t>>(_stream, _obj.dimensions, _format);
        xerus::misc::write_to_stream<std::vector<xerus::Tensor>>(_stream, _obj.components, _format);
    }


    void stream_reader(std::istream& _stream, BlockTT &_obj, const misc::FileFormat _format) {
        size_t ver = xerus::misc::read_from_stream<size_t>(_stream, _format);
        REQUIRE(ver == 1, "Unknown stream version to open (" << ver << ")");

        _obj.P = xerus::misc::read_from_stream<size_t>(_stream, _format);
        _obj.corePosition  = xerus::misc::read_from_stream<size_t>(_stream, _format);
        _obj.blockPosition = xerus::misc::read_from_stream<size_t>(_stream, _format);
        _obj.dimensions = xerus::misc::read_from_stream<std::vector<size_t>>(_stream, _format);
        _obj.components = xerus::misc::read_from_stream<std::vector<Tensor>>(_stream, _format);
    }


    bool BlockTT::all_entries_valid() const {
        for (const auto& comp : components) {
            if (! comp.all_entries_valid()) { return false; }
        }
        return true;
    }

} } // namespace xerus
