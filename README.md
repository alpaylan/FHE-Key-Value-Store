# Fully-Homomorphic Encryption

This repository is created as part of the [Bounty Program](https://github.com/zama-ai/bounty-program), as the solution to the bounty [Encrypted Key-Value Database](https://github.com/zama-ai/bounty-program/blob/main/Bounties/Engineering/create-key-value-database-app.md).

## Encrypted Key-Value Database Interface

The database interfaces three functions.

- Insert: Inserts a key-value pair into the database.
- Replace: Replaces the value of a key-value pair in the database.
- Query: Queries the value of a key-value pair in the database.

## Database Implementation

The database is implemented as a linear encrypted array of key-value pairs.

Each interface function is implemented as a fully-homomorphic encryption circuit. Values are encrypted, homomorphically processed, and decrypted.

For each operation, the database is iterated over, and the key-value pairs are processed. The key-value pairs are processed in chunks of size `CHUNK_SIZE`, with the default configuration `CHUNK_SIZE=4`.

## Implementation Details

This repository consists of 4 python files, each implementing a variation of the database.

- `KVStore.py`: The canonical implementation of the database.
    This implementation is the most clear, but is not optimized for performance.
    It uses a different circuit for each operation.
- `KVStore_commented.py`: The canonical implementation of the database, with comments.
- `KVStore_alternative.py`: An alternative implementation of the database.
    This implementation is more efficient than the canonical implementation, but is less clear.
    Hence, canonical implementation should be used for tutorial, teaching and learning purposes.
- `KVStore_4bit.py`: A 4-bit implementation of the database.
    This implementation is a simple, 4 bit key/value-width implementation of the database.
    It does not use encoding as there is only one chunk for each key/value. It operates directly
    on single 4 bit chunks.

## Usage

To use the database, first set the database configuration using global variables at the top of the code.

```python
# The number of entries in the database
NUMBER_OF_ENTRIES = 5
# The number of bits in each chunk
CHUNK_SIZE = 4

# The number of bits in the key and value
KEY_SIZE = 32
VALUE_SIZE = 32
```

As padding is not implemented, the key and value sizes must be a multiple of the chunk size.

`KEY_SIZE % CHUNK_SIZE == 0` and `VALUE_SIZE % CHUNK_SIZE == 0` must be true.

## Tutorial

There are two tutorials in this repository. One is this tutorial section, which is more concise and tries not to go into the details of the implementation as much as possible. The other is the `Tutorial.ipynb` file, which is an interactive tutorial that separates the canonical implementation into many small jupyter notebook cells, provides relevant information and allows the reader to tweak parameters easily.

This non-interactive tutorial will go over the commented canonical implementation of the database by neglecting some of the details of the implementation.

### Defining The State and Indexers

Firstly, we define the state of the database, and the indexers for each part of the state.

Indexers are used to extract bits from the state for flag, key and value.

State is defines with the following tensor shape:

| Flag Size | Key Size | Number of Key Chunks | Value Size | Number of Value Chunks |
| ---       | ---      | ---                  | ---        | ---                    |
| 1         | 32       | 32/4 = 8             | 32         | 32/4 = 8               |
| 1         | 8        | 8/4 = 2              | 16         | 16/4 = 4               |
| 1         | 4        | 4/4 = 1              | 4          | 4/4 = 1                |

The following code defines the state shape, and the indexers for each part of the state.

```python
# Required number of chunks to store keys and values
NUMBER_OF_KEY_CHUNKS = KEY_SIZE // CHUNK_SIZE
NUMBER_OF_VALUE_CHUNKS = VALUE_SIZE // CHUNK_SIZE

STATE_SHAPE = (NUMBER_OF_ENTRIES, 1 + NUMBER_OF_KEY_CHUNKS + NUMBER_OF_VALUE_CHUNKS)

# Indexers for each part of the state
FLAG = 0
KEY = slice(1, 1 + NUMBER_OF_KEY_CHUNKS)
VALUE = slice(1 + NUMBER_OF_KEY_CHUNKS, None)
```

### Defining Encode/Decode Functions

Encode/Decode functions are used to convert between integers and numpy arrays. The interface exposes integers, but the state is stored and processed as a numpy array.

Below are examples of the encode/decode functions.

#### Encode

| Function Call | Input(Integer) | Array-Width | Result(Numpy Array) |
| --- | --- | --- | --- |
| encode(25, 4) | 25 | 4 | [0, 0, 1, 9] |
| encode(40, 4) | 40 | 4 | [0, 0, 2, 8] |
| encode(11, 3) | 11 | 3 | [0, 0, 11] |

#### Decode

| Function Call | Input(Numpy Array) | Result(Integer) |
| --- | --- | --- |
| decode([0, 0, 1, 9]) | [0, 0, 1, 9] | 25 |
| decode([0, 0, 2, 8]) | [0, 0, 2, 8] | 40 |
| decode([0, 0, 11]) | [0, 0, 11] | 11 |

### Defining The Required Lookup Tables

The circuit implementations must use operations that are supported by the library. The library supports a limited set of operations, and lookup tables are used to implement the remaining operations.

One such operation is `if`, encrypted values cannot be used to control the flow of the circuit. Instead, the `if` operation is implemented using a lookup table.

The lookup table is used to implement the `keep_selected` function.

```python
def keep_selected(value, selected):
  if selected:
      return value
  else:
      return 0
```

```python
keep_selected_lut = cnp.LookupTable([0 for _ in range(16)] + [i for i in range(16)])
```

We can then encapsulate the function logic as given below.

```python
def keep_selected_with_tlu(value, selected):
  packed = (2**CHUNK_SIZE ) * selected + value
  return keep_selected_lut[packed]
```

Selected is pushed as the most significant bit of the value, and the lookup table is used to extract the value.

Below are some examples from the lookup table.

| Input(Value, Selected) | Lookup Table Input | Result(Value) |
| --- | --- | --- |
| (10, 1) | 26 | 10 |
| (10, 0) | 10 | 0 |
| (11, 1) | 27 | 11 |

As a concise function, we can also write it as:

`keep_selected_lut(i=0..15, 1) -> 0`

`keep_selected_lut(i=16..31, 0) -> i - 16`

### Defining The Circuit Implementation Functions

The circuit implementation functions are used to implement the key-value store circuits. Circuits are implemented using the set of operations supported by the library, and the lookup table defined above.

For more details, reader is advised to read the commented operation or follow the detailed interactive tutorial at the `Tutorial.ipynb` file to not clutter this tutorial. The commented code can be found in the `KVStore_commented.py` file.

### Defining The Database Interface

The database interface exposes 3 functions, and stores the state of the database along with the circuits used to implement the database.

```python
class KeyValueDatabase:
    """
    A key-value database that uses fully homomorphic encryption circuits to store the data.
    """

    # The state of the database, it holds all the 
    # keys and values as a table of entries
    _state: np.ndarray

    # The circuits used to implement the database
    _insert_circuit: cnp.Circuit
    _replace_circuit: cnp.Circuit
    _query_circuit: cnp.Circuit
```

### Defining The Input Sets

The input sets are used to initialize the circuits with the correct parameters. Two input sets are used depending on the inputs, one for binary inputs(query operation) and one for ternary inputs(insert and replace operations).

The input set for the query circuit:

```python
inputset_binary = [
    (
        np.zeros(STATE_SHAPE, dtype=np.int64), # state
        np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
    )
]
```

The input set for the insert and replace circuits:

```python
inputset_ternary = [
    (
        np.zeros(STATE_SHAPE, dtype=np.int64), # state
        np.ones(NUMBER_OF_KEY_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # key
        np.ones(NUMBER_OF_VALUE_CHUNKS, dtype=np.int64) * (2**CHUNK_SIZE - 1), # value
    )
]
```

### Defining The Compilers and Compiling The Circuits

For each circuit function, a compiler is defined.

The compiler is created by passing the circuit function, and the `encryptedness/plainness` of each variable. Encrypted variables are treated as homomorphic encrypted values, and plain variables are treated as plain values.

The compiler is then used to compile the circuit to use for the database operations.

```python
insert_compiler = cnp.Compiler(
    _insert_impl,
    {"state": "encrypted", "key": "encrypted", "value": "encrypted"}
)
self._insert_circuit = insert_compiler.compile(inputset_ternary, configuration)
```

### Using The Database

The database can be used to insert, replace, and query the database.

Database is initialize as follows:

```python
db = KeyValueDatabase()
```

The database can be used as follows:

```python
db.insert(3, 4)
assert db.query(3) == 4

db.replace(3, 1)
assert db.query(3) == 1

db.insert(25, 40)
assert db.query(25) == 40

assert db.query(4) == None

db.replace(3, 5)
assert db.query(3) == 5
```

Insert and Replace operations modify the internal state of the database, whereas query operation retrieves the value from the database, and returns `None` if value is not found.

### Running The Example

Once `concrete-numpy:0.8.0` is installed, the canonical example can be run using the following command:

```bash
python3 KVStore.py
```

### Important Notes: Virtual Circuits

As compiling the circuit is slow, readers may use `virtual circuits` to speed up the compilation process. Virtual circuits are, simply put, a way to simulate the circuit without actually compiling it. Throughout the implementation, I first worked on virtual circuits, and used actual circuit compilation to test correctness of the implementation.

Virtual circuits can be enabled by uncommenting the last line of the configuration as shown below:

```python
configuration = cnp.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",
    # virtual=True,
)
```

As virtual circuits cannot generate keys, you should also comment the `self._<operation>_circuit.keygen()` lines inside the `__init__` function.

Example of the keygen is below:

```python
self._insert_circuit.keygen()
```

## Conclusion

This tutorial introduced the key-value store example, and explained how to implement it using the library. For more detailed information, readers are advised to read the commented code in the `KVStore_commented.py` file.

Overall, I enjoyed working on this project and tutorial. FHE is a very interesting topic, and I believe that it will be used in many applications in the future. I hope that this tutorial will help readers to get started with the library, and I am looking forward to seeing what people will build with it.
