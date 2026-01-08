/*
 * opencog/atoms/aten/ATenSpace.cc
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <sstream>

#include <opencog/util/exceptions.h>
#include <opencog/atoms/aten/ATenSpace.h>
#include <opencog/atomspace/AtomSpace.h>

using namespace opencog;

// ============================================================
// Constructor / Destructor

ATenSpace::ATenSpace(AtomSpace* as, const std::string& device)
	: _device(device)
	, _requires_grad(false)
	, _atomspace(as)
	, _tensor_count(0)
	, _total_elements(0)
{
	if (_atomspace) {
		_tensor_key = _atomspace->add_node(PREDICATE_NODE, "*-TensorKey-*");
	}
}

ATenSpace::~ATenSpace()
{
	clear();
}

// ============================================================
// Configuration

void ATenSpace::set_atomspace(AtomSpace* as)
{
	std::lock_guard<std::mutex> lock(_mtx);
	_atomspace = as;
	if (_atomspace) {
		_tensor_key = _atomspace->add_node(PREDICATE_NODE, "*-TensorKey-*");
	} else {
		_tensor_key = Handle::UNDEFINED;
	}
}

void ATenSpace::set_device(const std::string& device)
{
	std::lock_guard<std::mutex> lock(_mtx);
	_device = device;
	// TODO: Move existing tensors to new device if using ATen
}

// ============================================================
// Named tensor management

void ATenSpace::register_tensor(const std::string& name,
                                const ATenValuePtr& tensor)
{
	std::lock_guard<std::mutex> lock(_mtx);

	// Check if replacing existing tensor
	auto it = _named_tensors.find(name);
	if (it != _named_tensors.end()) {
		_total_elements -= it->second->numel();
		_tensor_count--;
	}

	_named_tensors[name] = tensor;
	_tensor_count++;
	_total_elements += tensor->numel();
}

ATenValuePtr ATenSpace::get_tensor(const std::string& name) const
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _named_tensors.find(name);
	if (it != _named_tensors.end())
		return it->second;
	return nullptr;
}

bool ATenSpace::has_tensor(const std::string& name) const
{
	std::lock_guard<std::mutex> lock(_mtx);
	return _named_tensors.find(name) != _named_tensors.end();
}

void ATenSpace::remove_tensor(const std::string& name)
{
	std::lock_guard<std::mutex> lock(_mtx);

	auto it = _named_tensors.find(name);
	if (it != _named_tensors.end()) {
		_total_elements -= it->second->numel();
		_tensor_count--;
		_named_tensors.erase(it);
	}
}

std::vector<std::string> ATenSpace::get_tensor_names() const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::vector<std::string> names;
	names.reserve(_named_tensors.size());
	for (const auto& pair : _named_tensors)
		names.push_back(pair.first);
	return names;
}

void ATenSpace::clear()
{
	std::lock_guard<std::mutex> lock(_mtx);

	_named_tensors.clear();
	_tensor_count = 0;
	_total_elements = 0;
}

// ============================================================
// Tensor creation utilities

ATenValuePtr ATenSpace::zeros(const std::vector<int64_t>& shape,
                              const std::string& name)
{
	ATenValuePtr tensor = createATenZeros(shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

ATenValuePtr ATenSpace::ones(const std::vector<int64_t>& shape,
                             const std::string& name)
{
	ATenValuePtr tensor = createATenOnes(shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

ATenValuePtr ATenSpace::random(const std::vector<int64_t>& shape,
                               const std::string& name)
{
	ATenValuePtr tensor = createATenRandom(shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

ATenValuePtr ATenSpace::from_vector(const std::vector<double>& data,
                                    const std::vector<int64_t>& shape,
                                    const std::string& name)
{
	ATenValuePtr tensor = createATenFromVector(data, shape);

	if (!name.empty()) {
		register_tensor(name, tensor);
	}

	return tensor;
}

// ============================================================
// Atom-Tensor integration

void ATenSpace::attach_tensor(const Handle& atom, const ATenValuePtr& tensor)
{
	if (not _atomspace)
		throw RuntimeException(TRACE_INFO,
			"ATenSpace: no AtomSpace configured");

	_atomspace->set_value(atom, _tensor_key, tensor);
}

ATenValuePtr ATenSpace::get_attached_tensor(const Handle& atom) const
{
	if (not _tensor_key)
		return nullptr;

	ValuePtr val = atom->getValue(_tensor_key);
	return ATenValueCast(val);
}

bool ATenSpace::has_attached_tensor(const Handle& atom) const
{
	return get_attached_tensor(atom) != nullptr;
}

// ============================================================
// Statistics

std::string ATenSpace::to_string() const
{
	std::lock_guard<std::mutex> lock(_mtx);

	std::ostringstream oss;
	oss << "ATenSpace {\n";
	oss << "  device: " << _device << "\n";
	oss << "  requires_grad: " << (_requires_grad ? "true" : "false") << "\n";
	oss << "  tensor_count: " << _tensor_count << "\n";
	oss << "  total_elements: " << _total_elements << "\n";
	oss << "  registered_tensors: [\n";

	for (const auto& pair : _named_tensors) {
		auto shp = pair.second->shape();
		oss << "    " << pair.first << ": shape=(";
		for (size_t i = 0; i < shp.size(); i++) {
			if (i > 0) oss << ",";
			oss << shp[i];
		}
		oss << "), numel=" << pair.second->numel() << "\n";
	}

	oss << "  ]\n";
	oss << "}";

	return oss.str();
}
