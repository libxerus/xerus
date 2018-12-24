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
 * @brief Default include file for the xerus library.
 */

// NOTE must not be "pragma once" as that keyword is not supported in precompiled headers
#ifndef XERUS_H
#define XERUS_H

    #define MISC_NAMESPACE xerus
    
    // The misc stuff needed by xerus
    #include "xerus/misc.h"
    
    // File which sets the custom log levels
    #include "xerus/tensorLogger.h"

    // All the xerus headers
	#include "xerus/blasLapackWrapper.h"
    #include "xerus/index.h"
    #include "xerus/indexedTensorReadOnly.h"
    #include "xerus/indexedTensorWritable.h"
    #include "xerus/indexedTensor.h"
    #include "xerus/indexedTensorMoveable.h"
    #include "xerus/indexedTensorList.h"
    #include "xerus/tensor.h"
    #include "xerus/cholmod_wrapper.h"
    #include "xerus/sparseTimesFullContraction.h"
    #include "xerus/sparseTimesFullContraction.h"
    #include "xerus/indexedTensor_tensor_factorisations.h"
    #include "xerus/tensorNetwork.h"
    #include "xerus/contractionHeuristic.h"
    #include "xerus/ttNetwork.h"
    #include "xerus/htNetwork.h"
    #include "xerus/ttStack.h"
	#include "xerus/performanceData.h"
	#include "xerus/measurments.h"
    #include "xerus/algorithms/als.h"
    #include "xerus/algorithms/steepestDescent.h"
    #include "xerus/algorithms/cg.h"
    #include "xerus/algorithms/decompositionAls.h"
    #include "xerus/algorithms/adf.h"
    #include "xerus/algorithms/asd.h"
    #include "xerus/algorithms/iht.h"
    #include "xerus/algorithms/largestEntry.h"
    #include "xerus/algorithms/randomSVD.h"
    #include "xerus/applications/uq.h"
    #include "xerus/applications/uqAdf.h"
    
	#include "xerus/examples/specificLowRankTensors.h"

#endif
