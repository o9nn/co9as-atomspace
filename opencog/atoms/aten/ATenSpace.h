/*
 * opencog/atoms/aten/ATenSpace.h
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

#ifndef _OPENCOG_ATEN_SPACE_H
#define _OPENCOG_ATEN_SPACE_H

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <opencog/atoms/base/Handle.h>
#include <opencog/atoms/aten/ATenValue.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * ATenSpace provides a centralized manager for tensor values within
 * an AtomSpace context. It offers:
 *
 * - Named tensor registration and lookup
 * - Tensor computation graph management
 * - Device placement (CPU/GPU) coordination
 * - Gradient tracking for automatic differentiation
 * - Memory pool management for tensor allocation
 *
 * This is designed to integrate PyTorch-style tensor operations
 * with the AtomSpace hypergraph representation.
 */
class ATenSpace
{
private:
	// Thread safety
	mutable std::mutex _mtx;

	// Named tensor registry
	std::map<std::string, ATenValuePtr> _named_tensors;

	// Tensor key for attaching to atoms
	Handle _tensor_key;

	// Device configuration
	std::string _device; // "cpu" or "cuda:N"
	bool _requires_grad;

	// Reference to the AtomSpace (non-owning)
	AtomSpace* _atomspace;

	// Statistics
	size_t _tensor_count;
	size_t _total_elements;

public:
	/**
	 * Construct an ATenSpace.
	 *
	 * @param as Optional AtomSpace to associate with
	 * @param device Device for tensor allocation ("cpu" or "cuda:N")
	 */
	ATenSpace(AtomSpace* as = nullptr, const std::string& device = "cpu");
	~ATenSpace();

	// Prevent copying
	ATenSpace(const ATenSpace&) = delete;
	ATenSpace& operator=(const ATenSpace&) = delete;

	// ========================================
	// Configuration

	/**
	 * Set the associated AtomSpace.
	 */
	void set_atomspace(AtomSpace* as);
	AtomSpace* get_atomspace() const { return _atomspace; }

	/**
	 * Set the default device for tensor allocation.
	 * @param device "cpu" or "cuda:N"
	 */
	void set_device(const std::string& device);
	std::string get_device() const { return _device; }

	/**
	 * Enable/disable gradient tracking for new tensors.
	 */
	void set_requires_grad(bool req) { _requires_grad = req; }
	bool get_requires_grad() const { return _requires_grad; }

	// ========================================
	// Named tensor management

	/**
	 * Register a tensor with a name.
	 * @param name Unique identifier for the tensor
	 * @param tensor The tensor value to register
	 */
	void register_tensor(const std::string& name, const ATenValuePtr& tensor);

	/**
	 * Get a tensor by name.
	 * @param name The registered name
	 * @return The tensor, or nullptr if not found
	 */
	ATenValuePtr get_tensor(const std::string& name) const;

	/**
	 * Check if a tensor exists with the given name.
	 */
	bool has_tensor(const std::string& name) const;

	/**
	 * Remove a tensor by name.
	 */
	void remove_tensor(const std::string& name);

	/**
	 * Get all registered tensor names.
	 */
	std::vector<std::string> get_tensor_names() const;

	/**
	 * Clear all registered tensors.
	 */
	void clear();

	// ========================================
	// Tensor creation utilities

	/**
	 * Create a zeros tensor and optionally register it.
	 */
	ATenValuePtr zeros(const std::vector<int64_t>& shape,
	                   const std::string& name = "");

	/**
	 * Create a ones tensor and optionally register it.
	 */
	ATenValuePtr ones(const std::vector<int64_t>& shape,
	                  const std::string& name = "");

	/**
	 * Create a random tensor and optionally register it.
	 */
	ATenValuePtr random(const std::vector<int64_t>& shape,
	                    const std::string& name = "");

	/**
	 * Create a tensor from data and optionally register it.
	 */
	ATenValuePtr from_vector(const std::vector<double>& data,
	                         const std::vector<int64_t>& shape,
	                         const std::string& name = "");

	// ========================================
	// Atom-Tensor integration

	/**
	 * Attach a tensor to an atom using the default tensor key.
	 */
	void attach_tensor(const Handle& atom, const ATenValuePtr& tensor);

	/**
	 * Get a tensor attached to an atom.
	 */
	ATenValuePtr get_attached_tensor(const Handle& atom) const;

	/**
	 * Check if an atom has an attached tensor.
	 */
	bool has_attached_tensor(const Handle& atom) const;

	/**
	 * Get the default tensor key predicate.
	 */
	Handle get_tensor_key() const { return _tensor_key; }

	// ========================================
	// Statistics

	/**
	 * Get the number of registered tensors.
	 */
	size_t tensor_count() const { return _tensor_count; }

	/**
	 * Get the total number of elements across all tensors.
	 */
	size_t total_elements() const { return _total_elements; }

	/**
	 * Get a summary string of the ATenSpace state.
	 */
	std::string to_string() const;
};

typedef std::shared_ptr<ATenSpace> ATenSpacePtr;

/**
 * Create a new ATenSpace.
 */
template<typename ... Args>
ATenSpacePtr createATenSpace(Args&& ... args)
{
	return std::make_shared<ATenSpace>(std::forward<Args>(args)...);
}

/** @}*/
} // namespace opencog

#endif // _OPENCOG_ATEN_SPACE_H
