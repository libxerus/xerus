#include "misc.h"
#include<pybind11/numpy.h>


std::vector<size_t> strides_from_dimensions_and_item_size(const std::vector<size_t>& _dimensions, const size_t _item_size) {
	const size_t ndim = _dimensions.size();
	std::vector<size_t> strides(ndim, 0);
	if (ndim > 0) {
		strides[ndim-1] = _item_size;
		for (size_t i=ndim-1; i>0; --i) {
			strides[i-1] = _dimensions[i] * strides[i];
		}
	}
	return strides;
}

Tensor Tensor_from_buffer(buffer& _b) {
	// cast buffer into c_contiguous array (removes boilerplate code)
	auto b = array_t<value_t, array::c_style | array::forcecast>::ensure(_b);
	buffer_info info = b.request();

	if (info.shape.size() == 1 and info.shape[0] == 0) {
		return Tensor({}, Tensor::Representation::Dense, Tensor::Initialisation::None);
	}

	std::vector<size_t> dimensions(info.shape.begin(), info.shape.end());
	Tensor result(dimensions, Tensor::Representation::Dense, Tensor::Initialisation::None);
	misc::copy(result.get_unsanitized_dense_data(), static_cast<double*>(info.ptr), result.size);
	return result;
}

void expose_tensor(module& m) {
	enum_<Tensor::Representation>(m, "Representation", "Possible representations of Tensor objects.")
		.value("Dense", Tensor::Representation::Dense)
		.value("Sparse", Tensor::Representation::Sparse)
	;
	enum_<Tensor::Initialisation>(m, "Initialisation", "Possible initialisations of new Tensor objects.")
		.value("Zero", Tensor::Initialisation::Zero)
		.value("Uninitialized", Tensor::Initialisation::None)  /* None is a protected keyword in python */
	;

	class_<Tensor>(m, "Tensor", "a non-decomposed Tensor in either sparse or dense representation", buffer_protocol())
	// .def_buffer([](Tensor& t) -> buffer_info {
	// 	std::cerr << "buffer" << std::endl;
	// 	return buffer_info(
	// 		t.get_dense_data(),                    /* Pointer to buffer */
	// 		sizeof(value_t),                       /* Size of one scalar */
	// 		format_descriptor<value_t>::format(),  /* Python struct-style format descriptor */
	// 		t.order(),                             /* Number of dimensions */
	// 		t.dimensions,                          /* Buffer dimensions */
	// 		strides_from_dimensions_and_item_size(t.dimensions, sizeof(value_t))  /* Strides (in bytes) for each index */
	// 	);
	// })
	.def(pickle(
		[](const Tensor &_self) { // __getstate__
			return bytes(misc::serialize(_self));
		},
		[](bytes _bytes) { // __setstate__
			return misc::deserialize<Tensor>(_bytes);
		}
	))
	.def(init<>(), "constructs an empty Tensor")
	.def(init<Tensor>())
	.def(init<TensorNetwork>())
	.def(init<Tensor::DimensionTuple, Tensor::Representation, Tensor::Initialisation>(),
		"constructs a Tensor with the given dimensions",
		arg("dim"),
		arg("repr")=Tensor::Representation::Sparse,
		arg("init")=Tensor::Initialisation::Zero
	)
	.def(init<Tensor::DimensionTuple, std::function<value_t(std::vector<size_t>)>>())
	.def_static("from_function", +[](const Tensor::DimensionTuple& _dim, const std::function<value_t(std::vector<size_t>)> _f){
		LOG(warning, "Deprecation warning: `from_function` is deprecated and will be removed in Xerus v5.0.0. Use the `Tensor` constructor instead.");
		return Tensor(_dim, _f);
	})
	.def_static("from_buffer", &Tensor_from_buffer)
	.def_static("from_ndarray", &Tensor_from_buffer)
	.def("to_ndarray", +[](const xerus::Tensor& _self){
		static_assert(std::is_same<value_t, double>::value);
		Tensor* shcp = new xerus::Tensor(_self);  // create a shallow copy (this effectively increases its reference count for the required amount of time)
		shcp->ensure_own_data_and_apply_factor();
		shcp->use_dense_representation();
		capsule cps(shcp, [](void *v) { delete reinterpret_cast<xerus::Tensor*>(v); });
		return array(dtype::of<double>(), shcp->dimensions, shcp->get_unsanitized_dense_data(), cps);
	})
	.def_property_readonly("dimensions", +[](Tensor &_A) {
		return _A.dimensions;
	})
	.def("degree", &Tensor::degree)
	.def("order", &Tensor::order)
	.def_readonly("factor", &Tensor::factor)
	.def_readonly("size", &Tensor::size)
	.def("one_norm", &Tensor::one_norm)
	.def("frob_norm", &Tensor::frob_norm)
	.def_static("random",
		+[](std::vector<size_t> _dim) {
		return xerus::Tensor::random(_dim);
		},
		"Construct a tensor with i.i.d. Gaussian random entries."
		parametersDocstr
		"dim : list or tuple of int\n"
		"n : list or tuple of int, optional\n"
		"    number of non-zero entries",
		arg("dim")
		)
	.def_static("random",
		+[](std::vector<size_t> _dim, size_t _n) {
		return xerus::Tensor::random(_dim, _n);
		},
		arg("dim"), arg("n")
		)
	.def_static("random_orthogonal",
		+[](std::vector<size_t> _dimLhs, std::vector<size_t> _dimRhs) {
		return xerus::Tensor::random_orthogonal(_dimLhs, _dimRhs);
		})
	.def_static("ones", &Tensor::ones,
		 "Constructs a tensor of given dimensions that is equal to 1 everywhere."
		  parametersDocstr "dim : list or tuple of int",
	arg("dim")
	)
	.def_static("identity", &Tensor::identity, 
		"Constructs a Tensor representation of the identity operator with the given dimensions."
		parametersDocstr "dim : list or tuple of int",
arg("dim")
	)
	.def_static("kronecker", &Tensor::kronecker, 
		"Constructs a Tensor representation of the kronecker delta (=1 where all indices are identical, =0 otherwise)."
		parametersDocstr "dim : list or tuple of int",
arg("dim")
	)
	.def_static("dirac", static_cast<Tensor (*)(Tensor::DimensionTuple, const Tensor::MultiIndex&)>(&Tensor::dirac),
		"Construct a Tensor with a single entry equals one and all other zero."
		parametersDocstr
		"dim : list or tuple of int\n"
		"pos : list or tuple of int\n"
		"    position of the 1 entry",
		arg("dim"), arg("pos")
	)
	.def_static("dirac", static_cast<Tensor (*)(Tensor::DimensionTuple, const size_t)>(&Tensor::dirac))
	.def("has_factor", &Tensor::has_factor)
	.def("is_dense", &Tensor::is_dense)
	.def("is_sparse", &Tensor::is_sparse)
	.def("sparsity", &Tensor::sparsity)
	.def("all_entries_valid", &Tensor::all_entries_valid)
	.def("reorder_cost", &Tensor::reorder_cost)
	.def("reinterpret_dimensions", &Tensor::reinterpret_dimensions,
		arg("dim"),
		"Reinterprets the dimensions of the tensor."
		parametersDocstr
		"dim : list or tuple of int"
	)
	.def("resize_mode", &Tensor::resize_mode,
		"Resizes a specific mode of the Tensor."
		parametersDocstr
		"mode : int\n"
		"newDim : int\n"
		"cutPos : int, optional (default: infinity)\n"
		"    The position within the selected mode in front of which slates are inserted or removed.",
		arg("mode"), arg("newDim"), arg("cutPos")=~0ul
	)
	.def("fix_mode", &Tensor::fix_mode,
		"Fixes a specific mode to a specific value, effectively reducing the order by one."
		parametersDocstr
		"mode : int\n"
		"value : int",
		arg("mode"), arg("value")
	)
	.def("remove_slate", &Tensor::remove_slate,
		"Removes a single slate from the Tensor, reducing dimension[mode] by one."
		parametersDocstr
		"mode : int\n"
		"pos : int",
		arg("mode"), arg("pos")
	)
	.def("perform_trace", &Tensor::perform_trace)
	.def("offset_add", &Tensor::offset_add)
	.def("use_dense_representation", &Tensor::use_dense_representation)
	.def("use_sparse_representation", &Tensor::use_sparse_representation,
		arg("epsilon")=EPSILON
	)
	.def("sparse_copy", &Tensor::sparse_copy)
	.def("dense_copy", &Tensor::dense_copy)
	.def("ensure_own_data", &Tensor::ensure_own_data)
	.def("ensure_own_data_no_copy", &Tensor::ensure_own_data_no_copy)
	.def("apply_factor", &Tensor::apply_factor)
	.def("ensure_own_data_and_apply_factor", &Tensor::ensure_own_data_and_apply_factor)
	.def_static("multiIndex_to_position", &Tensor::multiIndex_to_position)
	.def_static("position_to_multiIndex", &Tensor::position_to_multiIndex)
	/* .def("__call__", +[](Tensor *_this, const std::vector<Index> &_idx){ */
	/*     return  new xerus::internal::IndexedTensor<Tensor>(std::move((*_this)(_idx))); */
	/* }, keep_alive<0,1>(), return_value_policy::take_ownership ) */
	.def("__call__", +[](Tensor& _this, args _args){
		std::vector<Index> idx;
		idx.reserve(_args.size());
		for (size_t i=0; i<_args.size(); ++i) {
			  idx.push_back(*(_args[i].cast<Index *>()));
		}
		return new xerus::internal::IndexedTensor<Tensor>(std::move(_this(idx)));
	}, keep_alive<0,1>(), return_value_policy::take_ownership )
	.def("__str__", &Tensor::to_string)
	/* .def(-self) */
	.def("__neg__",
		+[](Tensor& _self) {
			return (-1)*_self;
		})
	.def(self + self)
	.def(self - self)
	.def(self += self)
	.def(self -= self)
	.def(self * value_t())
	.def(value_t() * self)
	.def(self *= value_t())
	.def(self / value_t())
	/* .def(self /= self) */
	.def("__itruediv__",
		+[](Tensor& _self, const value_t _other) {
			return (_self *= (1/_other));
		})

	.def("__getitem__", +[](Tensor &_this, size_t _i) {
		if (_i >= _this.size) {
			throw index_error("Index out of range");
		}
		return _this[_i];
	})
	.def("__getitem__", +[](Tensor &_this, std::vector<size_t> _idx) {
		return _this[_idx];
	})
	.def("__setitem__", +[](Tensor &_this, size_t _i, value_t _val) {
		_this[_i] = _val;
	})
	.def("__setitem__", +[](Tensor &_this, std::vector<size_t> _i, value_t _val) {
		_this[_i] = _val;
	})
	.def("__float__", [](const Tensor &_self){
			if (_self.order() != 0) {
				throw value_error("order must be 0");
			}
			return value_t(_self());
	})
	;
}

// NOTE The following code (when defined globally) would cast every xerus::Tensor to a numpy.ndarray.
//      This would allow for cleaner code like the following:
//          tt = xe.TTTensor([3])
//          tt.set_component(0, np.arange(3)[None,:,None])
/* namespace pybind11 { namespace detail { */
/*   template <> struct type_caster<Tensor> */
/*   { */
/*     public: */

/*       PYBIND11_TYPE_CASTER(Tensor, _("Tensor")); */

/*       // Conversion part 1 (Python -> C++) */
/*       bool load(handle src, bool convert) */
/*       { */
/*         if ( !convert and !array_t<value_t>::check_(src) ) */
/*           return false; */

/*         auto buf = array_t<value_t, array::c_style | array::forcecast>::ensure(src); */
/*         if ( !buf ) */
/*           return false; */

/*         try { */
/*             value = Tensor_from_buffer(buf); */
/*         } catch (const std::runtime_error&) { */
/*             return false; */
/*         } */
/*         return true; */
/*       } */

/*       //Conversion part 2 (C++ -> Python) */
/*       static handle cast(const Tensor& src, return_value_policy policy, handle parent) */
/*       { */
/*         std::cerr << "cast Tensor -> array" << std::endl; */
/*         std::cerr << "    create dimension vector" << std::endl; */
/*         std::vector<size_t> shape = src.dimensions; */
/*         std::cerr << "    create strides vector" << std::endl; */
/*         std::vector<size_t> strides = strides_from_dimensions_and_item_size(shape, sizeof(value_t)); */

/*         /1* array a(std::move(shape), std::move(strides), src.get_dense_data()); *1/ */
/*         /1* return a.release(); *1/ */
/*         if (src.is_dense()) { */
/*             std::cerr << "    is_dense" << std::endl; */
/*             array a(std::move(shape), std::move(strides), src.get_unsanitized_dense_data()); */
/*             return a.release(); */
/*         } else { */
/*             std::cerr << "    is_sparse" << std::endl; */
/*             Tensor tmp(src); */
/*             array a(std::move(shape), std::move(strides), tmp.get_dense_data()); */
/*             return a.release(); */
/*         } */
/*       } */
/*   }; */
/* }} // namespace pybind11::detail */
