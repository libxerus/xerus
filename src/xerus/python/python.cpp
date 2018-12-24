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
 * @brief Definition of the python bindings.
 */

#include "misc.h"

// TODO get_copy() wrapper in python correct manner
BOOST_PYTHON_MODULE(xerus) {
    using namespace xerus;

    import_array(); // for numpy

    bool show_user_defined = true;
    bool show_signatures = false;
    docstring_options doc_options(show_user_defined, show_signatures);

    // xerus version
    scope().attr("VERSION_MAJOR") = VERSION_MAJOR;
    scope().attr("VERSION_MINOR") = VERSION_MINOR;
    scope().attr("VERSION_REVISION") = VERSION_REVISION;
    scope().attr("VERSION_COMMIT") = VERSION_COMMIT;

    VECTOR_TO_PY(size_t, "IntegerVector");
    VECTOR_TO_PY(double, "DoubleVector");

    expose_indexedTensors();
    expose_factorizations();
    expose_tensor();
    expose_tensorNetwork();
    expose_ttnetwork();
    expose_htnetwork();

    expose_leastSquaresAlgorithms();
    expose_recoveryAlgorithms();

    expose_misc();
}
