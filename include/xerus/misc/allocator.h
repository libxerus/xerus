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
 * @brief Header file for the data structures used by the custom new and delete operators
 */


#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <set>
#include <ext/malloc_allocator.h>

namespace xerus { namespace misc {

extern void* (*r_malloc)(size_t);
extern void (*r_free)(void*);
	
using Mallocator = __gnu_cxx::malloc_allocator<void*>;

struct AllocatorStorage {
	static constexpr size_t POOL_SIZE = 3*1024*1024;
	static constexpr size_t BUCKET_SIZE = 32;
	static constexpr size_t NUM_BUCKETS = 64;
	static constexpr size_t SMALLEST_NOT_CACHED_SIZE = BUCKET_SIZE * NUM_BUCKETS;
	
	std::array<std::vector<uint8_t*, Mallocator>, NUM_BUCKETS> buckets;
	std::vector<std::pair<uint8_t*, uint8_t*>, Mallocator> pools;
	
	AllocatorStorage();
	~AllocatorStorage();
	
	static void create_new_pool();

	static unsigned long allocCount[NUM_BUCKETS];
	static long maxAlloc[NUM_BUCKETS];
	static long currAlloc[NUM_BUCKETS];
};

extern thread_local AllocatorStorage astore;

}}



