#include "misc.h"
using namespace internal;

void expose_misc(module& m) {
	m.def("frob_norm", +[](const Tensor& _x){ return _x.frob_norm(); });
	m.def("frob_norm", +[](const TensorNetwork& _x){ return _x.frob_norm(); });
	m.def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<Tensor>&)>(&frob_norm));
	m.def("frob_norm", static_cast<value_t (*)(const IndexedTensorReadOnly<TensorNetwork>&)>(&frob_norm));

	m.def("approx_equal", static_cast<bool (*)(const TensorNetwork&, const TensorNetwork&, double)>(&approx_equal));
	m.def("approx_equal", static_cast<bool (*)(const Tensor&, const TensorNetwork&, double)>(&approx_equal));
	m.def("approx_equal", static_cast<bool (*)(const TensorNetwork&, const Tensor&, double)>(&approx_equal));
	m.def("approx_equal", static_cast<bool (*)(const Tensor&, const Tensor&, double)>(&approx_equal));
	m.def("approx_equal", +[](const Tensor& _l, const Tensor& _r) {
		return approx_equal(_l, _r);
	});
	m.def("approx_equal", +[](const Tensor& _l, const TensorNetwork& _r) {
		return approx_equal(_l, _r);
	});
	m.def("approx_equal", +[](const TensorNetwork& _l, const Tensor& _r) {
		return approx_equal(_l, _r);
	});
	m.def("approx_equal", +[](const TensorNetwork& _l, const TensorNetwork& _r) {
		return approx_equal(_l, _r);
	});

	m.def("log", +[](std::string _msg){
		LOG_SHORT(info, _msg);
	});

	enum_<misc::FileFormat>(m, "FileFormat")
		.value("BINARY", misc::FileFormat::BINARY)
		.value("TSV", misc::FileFormat::TSV)
	;

	/* m.def("save_to_file", static_cast<void (*)(const Tensor&)>(&misc::save_to_file), arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY); */
	/* m.def("save_to_file", static_cast<void (*)(const TTTensor&)>(&misc::save_to_file), arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY); */
	/* m.def("save_to_file", static_cast<void (*)(const TTOperator&)>(&misc::save_to_file), arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY); */
	/* m.def("save_to_file", static_cast<void (*)(const TensorNetwork&)>(&misc::save_to_file), arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY); */
	m.def("save_to_file", +[](const Tensor &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY);

	m.def("save_to_file", +[](const TensorNetwork &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY);

	m.def("save_to_file", +[](const TTTensor &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY);

	m.def("save_to_file", +[](const TTOperator &_obj, const std::string &_filename, misc::FileFormat _format){
		misc::save_to_file(_obj, _filename, _format);
	}, arg("object"), arg("filename"), arg("format")=misc::FileFormat::BINARY);

	m.def("load_from_file", +[](std::string _filename){
		// determine type stored in the file
		std::ifstream in(_filename);
		if (!in) {
			throw std::runtime_error("could not read file '" + _filename + "'");
		}
		std::string classname;
		in >> classname; // "Xerus"
		in >> classname;
		in.close();
		if (classname == "xerus::Tensor") {
			return cast(misc::load_from_file<Tensor>(_filename));
		}
		if (classname == "xerus::TensorNetwork") {
			return cast(misc::load_from_file<TensorNetwork>(_filename));
		}
		if (classname == "xerus::TTNetwork<false>") {
			return cast(misc::load_from_file<TTTensor>(_filename));
		}
		if (classname == "xerus::TTNetwork<true>") {
			return cast(misc::load_from_file<TTOperator>(_filename));
		}
		throw value_error("unknown class type '" + classname + "' in file '" + _filename + "'");  //TODO...
	});

	m.def("serialize", +[](const Tensor &_obj){ return bytes(misc::serialize(_obj)); }, arg("object"));
	m.def("serialize", +[](const TTTensor &_obj){ return bytes(misc::serialize(_obj)); }, arg("object"));
	m.def("serialize", +[](const TTOperator &_obj){ return bytes(misc::serialize(_obj)); }, arg("object"));
	m.def("serialize", +[](const HTTensor &_obj){ return bytes(misc::serialize(_obj)); }, arg("object"));
	m.def("serialize", +[](const HTOperator &_obj){ return bytes(misc::serialize(_obj)); }, arg("object"));
	m.def("serialize", +[](const TensorNetwork &_obj){ return bytes(misc::serialize(_obj)); }, arg("object"));

	m.def("deserialize", +[](std::string _bytes){
		// determine type stored in the file
		std::string classname = _bytes.substr(6, _bytes.find("\n"));  // 6 == "Xerus ".length()
		classname = classname.substr(0, classname.find(" "));
		if (classname == "xerus::Tensor") {
			return cast(misc::deserialize<Tensor>(_bytes));
		}
		if (classname == "xerus::TensorNetwork") {
			return cast(misc::deserialize<TensorNetwork>(_bytes));
		}
		if (classname == "xerus::TTNetwork<false>") {
			return cast(misc::deserialize<TTTensor>(_bytes));
		}
		if (classname == "xerus::TTNetwork<true>") {
			return cast(misc::deserialize<TTOperator>(_bytes));
		}
		throw value_error("unknown class type '" + classname + "'");  //TODO...
	});

	m.def("get_call_stack", &misc::get_call_stack);

	// translate all exceptions thrown inside xerus to own python exception class
	register_exception<misc::generic_error>(m, "xerus_error");  // xerus.generic_error does not work because of the '.'
}
