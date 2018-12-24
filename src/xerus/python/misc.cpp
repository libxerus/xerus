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
 * @brief Definition of the python bindings of save and restore, exceptions etc..
 */


#define NO_IMPORT_ARRAY
#include "misc.h"



void variable_argument_member_to_tuple_wrapper(const std::string &_name, const std::string &_tmpName) {
	boost::python::str pyCode;
	pyCode += "def patch_call_fn():\n";
	pyCode += std::string("  original = ")+_name+"\n";
	pyCode += std::string("  def ")+_tmpName+"( this, *args ):\n";
	pyCode += "    return original( this, args )\n";
	pyCode += std::string("  return ")+_tmpName+"\n";
	pyCode += _name + " = patch_call_fn()\n";
	boost::python::exec(pyCode, scope().attr("__dict__"));
}


using namespace internal;

void expose_misc() {
	def("frob_norm", +[](const Tensor& _x){ return _x.frob_norm(); });
	def("frob_norm", +[](const TensorNetwork& _x){ return _x.frob_norm(); });
	def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<Tensor>&)>(&frob_norm));
	def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<TensorNetwork>&)>(&frob_norm));

	def("approx_equal", static_cast<bool (*)(const TensorNetwork&, const TensorNetwork&, double)>(&approx_equal));
	def("approx_equal", static_cast<bool (*)(const Tensor&, const TensorNetwork&, double)>(&approx_equal));
	def("approx_equal", static_cast<bool (*)(const TensorNetwork&, const Tensor&, double)>(&approx_equal));
	def("approx_equal", static_cast<bool (*)(const Tensor&, const Tensor&, double)>(&approx_equal));
	def("approx_equal", +[](const Tensor& _l, const Tensor& _r) {
		return approx_equal(_l, _r);
	});
	def("approx_equal", +[](const Tensor& _l, const TensorNetwork& _r) {
		return approx_equal(_l, _r);
	});
	def("approx_equal", +[](const TensorNetwork& _l, const Tensor& _r) {
		return approx_equal(_l, _r);
	});
	def("approx_equal", +[](const TensorNetwork& _l, const TensorNetwork& _r) {
		return approx_equal(_l, _r);
	});

	def("log", +[](std::string _msg){
		LOG_SHORT(info, _msg);
	});

	enum_<misc::FileFormat>("FileFormat")
		.value("BINARY", misc::FileFormat::BINARY)
		.value("TSV", misc::FileFormat::TSV)
	;

	def("save_to_file", +[](const Tensor &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, (arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY) );

	def("save_to_file", +[](const TensorNetwork &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, (arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY) );

	def("save_to_file", +[](const TTTensor &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, (arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY) );

	def("save_to_file", +[](const TTOperator &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, (arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY) );

	def("load_from_file", +[](std::string _filename){
		// determine type stored in the file
		std::ifstream in(_filename);
		if (!in) {
			return object();
		}
		std::string classname;
		in >> classname; // "Xerus"
		in >> classname;
		in.close();
		if (classname == "xerus::Tensor") {
			return object(misc::load_from_file<Tensor>(_filename));
		}
		if (classname == "xerus::TensorNetwork") {
			return object(misc::load_from_file<TensorNetwork>(_filename));
		}
		if (classname == "xerus::TTNetwork<false>") {
			return object(misc::load_from_file<TTTensor>(_filename));
		}
		if (classname == "xerus::TTNetwork<true>") {
			return object(misc::load_from_file<TTOperator>(_filename));
		}
		LOG_SHORT(warning, "unknown class type '" << classname << "' in file '" << _filename << "'");
		return object();
	});

	// identity returns the cpp name to a python object
// 	def("identity", identity_);

	def("xethrow", +[](){XERUS_THROW(misc::generic_error() << misc::get_call_stack());});

	// translate all exceptions thrown inside xerus to own python exception class
	static char fully_qualified_gen_error_name[] = "xerus.generic_error";
	static PyObject* py_gen_error = PyErr_NewException(fully_qualified_gen_error_name, PyExc_Exception, 0);
	py::scope().attr("generic_error") = py::handle<>(py::borrowed(py_gen_error));

	register_exception_translator<misc::generic_error>([](const misc::generic_error &_e){
		LOG(pydebug, "custom exception handler called with " << _e.what());
		PyErr_SetString(py_gen_error, _e.what());
	});
}
