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
 * @brief Implementation of code coverage functions.
 */

#include <xerus/misc/codeCoverage.h>

#include <iostream>
#include <iomanip>
#include <chrono>
#include <signal.h>
#include <sstream>
#include <string>
#include <unordered_set>

#include <string.h> // for strsignal
#include <sys/stat.h>
#include <sys/mman.h> // For mlockall

#include <xerus/misc/standard.h>
#include <xerus/misc/exceptions.h>
#include <xerus/misc/stringUtilities.h>
#include <xerus/misc/fileUtilities.h>
#include <xerus/misc/random.h>
#include <xerus/misc/internal.h>

namespace xerus { namespace misc { namespace CodeCoverage {
	
	std::map<std::string, std::unordered_set<std::string>>* testsCovered;
	std::map<std::string, std::unordered_set<std::string>>* testsRequiredInit;
	
	
	void register_test(const std::string& _location, const std::string& _identifier) {
		// Initialization order is undefined, therefore the first one has to initialize.
		if(!testsRequiredInit) { testsRequiredInit = new std::map<std::string, std::unordered_set<std::string>>; }
		
		(*testsRequiredInit)[_location].insert(_identifier);
	}
	
	void covered(const std::string& _location, const std::string& _identifier) {
		// Initialization order is undefined, therefore the first covered test has to initialize.
		if(!testsCovered) { testsCovered = new std::map<std::string, std::unordered_set<std::string>>; }
		
		(*testsCovered)[_location].insert(_identifier);
	}
		
	void print_code_coverage() {
		const std::string unknownName("Function name not detected");
		
		// Set the required test from init
		
		// requiredTests[file][lineNumber][functionIdentifier]
		std::map<std::string, std::map<size_t, std::map<std::string, bool>>> requiredTests;
		
		for(const auto& localTests : *testsRequiredInit) {
			const auto locationParts = misc::explode(localTests.first, ':');
			REQUIRE(locationParts.size() == 2, "Error parsing the required tests.");
			const auto file = xerus::misc::normalize_pathname(locationParts[0]);
			const auto lineNumber = xerus::misc::from_string<size_t>(locationParts[1]);
			for(const auto& identifier : localTests.second) {
				requiredTests[file][lineNumber][identifier] = false;
			}
		}
		
		// Load further required tests
		const auto testLines = xerus::misc::explode(xerus::misc::read_file("build/required_tests.txt"), '\n');
		
		for(const auto& line : testLines) {
			const auto lineParts = xerus::misc::explode(line, '@');
			REQUIRE(lineParts.size() == 3, "Error parsing the required tests.");
			const auto locationParts = misc::explode(lineParts[1], ':');
			REQUIRE(locationParts.size() == 2, "Error parsing the required tests.");
			const auto file = xerus::misc::normalize_pathname(locationParts[0]);
			const auto lineNumber = xerus::misc::from_string<size_t>(locationParts[1]);
			if(requiredTests[file].count(lineNumber) == 0) {
				requiredTests[file][lineNumber][unknownName] = false;
			}
		} 
		
		// Assing covered tests to required ones
		for( const auto &test : *testsCovered ) {
			const auto locationParts = misc::explode(test.first, ':');
			REQUIRE(locationParts.size() == 2, "Error parsing the required tests.");
			const auto file = xerus::misc::normalize_pathname(locationParts[0]);
			const auto lineNumber = xerus::misc::from_string<size_t>(locationParts[1]);
			for( const auto& identifier : test.second ) {
				REQUIRE(requiredTests.count(file) > 0 && requiredTests.at(file).count(lineNumber) > 0 && (requiredTests.at(file).at(lineNumber).count(identifier) > 0 || requiredTests.at(file).at(lineNumber).count(unknownName) > 0), "Test for: " << file << ":" << lineNumber << "( " << identifier << " ) is not required???");
				if(requiredTests.at(file).at(lineNumber).count(unknownName) > 0) {
					requiredTests[file][lineNumber][unknownName] = true;
				} else {
					requiredTests[file][lineNumber][identifier] = true;
				}
			}
		}
		
		// Check which tests are covered and assamble per file statistics
		std::map<std::string, std::pair<size_t, size_t>> perFileStats;
		for( const auto &fileTests : requiredTests ) {
			for( const auto &lineTests : fileTests.second ) {
				for( const auto &test : lineTests.second ) {
					perFileStats[fileTests.first].first++;
					if(test.second) {
						perFileStats[fileTests.first].second++;
					} else {
						std::cout << "\033[1;31m Missing test for \033[0m" << fileTests.first << ":" << lineTests.first << ": " << test.first << std::endl;
					}
				}
			}
		}
		
		// Print statistics
		uint64_t totalExisting=0, totalPerformed=0;
		for ( const auto &file : perFileStats ) {
			const std::pair<size_t, size_t>& fstats = file.second;
			totalExisting += fstats.first;
			totalPerformed += fstats.second;
			if (fstats.first == fstats.second) {
				std::cout << "File " << file.first << " :\033[1;32m " << fstats.second << " of " << fstats.first << " tests performed\033[0m" << std::endl;
			} else {
				std::cout << "File " << file.first << " :\033[1;31m " << fstats.second << " of " << fstats.first << " tests performed\033[0m" << std::endl;
			}
		}
		
		std::cout << "In total: " << totalPerformed << " of " << totalExisting << " = " << 100*double(totalPerformed)/double(totalExisting) << "% covered" << std::endl;
	}

}}} // namespaces
