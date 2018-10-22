#pragma once

/* A fixed size array that knows how many items it's currently holding. */
template<class T>
class VariableArray {
public:
	VariableArray(int size) : _size(0), _capacity(size) { items = new T[_capacity]; }
	~VariableArray() { delete[] items; }

	T* items;

	T& operator[](size_t idx) { return items[idx]; }

	void set_size(size_t size) { _size = size; }
	size_t size() const { return _size; }

	size_t capacity() const { return _capacity; }
private:
	size_t _size;
	size_t _capacity;
};